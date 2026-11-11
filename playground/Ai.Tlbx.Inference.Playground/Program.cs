using System.Text.Json;
using Ai.Tlbx.Inference;
using Ai.Tlbx.Inference.Configuration;

var builder = WebApplication.CreateBuilder(args);
var apiKey = builder.Configuration["OPENAI_API_KEY"];
if (string.IsNullOrWhiteSpace(apiKey))
    throw new InvalidOperationException("Set OPENAI_API_KEY before starting the playground.");
builder.Services.AddAiInference(options => options.AddOpenAi(apiKey));
var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/health", () => Results.Ok(new { status = "ok" }));

app.MapPost("/api/chat", async (ChatRequest request, IAiInferenceClient client, CancellationToken ct) =>
{
    if (!Enum.TryParse<AiModel>(request.Model, true, out var model) ||
        !Enum.IsDefined(model) || model.GetProvider() != ProviderType.OpenAi)
        return Results.BadRequest(new { error = "Select a supported OpenAI model." });
    if (request.Messages is null || request.Messages.Count == 0 || request.Messages.Any(m =>
            m.Role is not ("user" or "assistant") || string.IsNullOrWhiteSpace(m.Content)))
        return Results.BadRequest(new { error = "Provide non-empty user and assistant messages." });
    var result = await client.CompleteAsync(new CompletionRequest
    {
        Model = model,
        SystemMessage = request.SystemMessage,
        Messages = request.Messages.Select(m => new ChatMessage
        {
            Role = m.Role == "user" ? ChatRole.User : ChatRole.Assistant,
            Content = m.Content
        }).ToArray()
    }, ct);
    return Results.Ok(new { content = result.Content, usage = result.Usage, model = result.Model.ToString(),
        tokenCostUsd = result.Usage.EstimateCost(model).ProviderCost });
});

app.MapPost("/api/agents", async (CreateAgentRequest request, IAiInferenceClient client, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(request.Model)) return Results.BadRequest(new { error = "Model is required." });
    if (request.McpUrl is not null && (!Uri.TryCreate(request.McpUrl, UriKind.Absolute, out var uri) || uri.Scheme != "https" || string.IsNullOrWhiteSpace(request.McpLabel)))
        return Results.BadRequest(new { error = "MCP requires an HTTPS URL and label." });
    var servers = request.McpUrl is null ? [] : new[] { new HostedAgentMcpServer
    {
        Label = request.McpLabel!, Url = new Uri(request.McpUrl),
        AllowedTools = request.McpAllowedTools?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    } };
    var session = await client.CreateHostedAgentSessionAsync(new HostedAgentOptions
    {
        Model = request.Model,
        Instructions = request.Instructions,
        ReasoningEffort = request.ReasoningEffort,
        EnableWebSearch = request.EnableWebSearch,
        AllowNetworkAccess = request.AllowNetworkAccess,
        McpServers = servers
    }, ct);
    return Results.Ok(session);
});

app.MapGet("/api/agents/{id}", async (string id, IAiInferenceClient client, CancellationToken ct) =>
    Results.Ok(await client.GetHostedAgentSessionAsync(id, ct)));
app.MapGet("/api/agents/{id}/items", async (string id, IAiInferenceClient client, CancellationToken ct) =>
    Results.Ok(await client.ListHostedAgentItemsAsync(id, ct)));
app.MapGet("/api/agents/{id}/usage", async (string id, string model, IAiInferenceClient client, CancellationToken ct) =>
{
    var hasRates = AgentRates.TryGetValue(model, out var rates);
    var turns = await client.ListHostedAgentTurnsAsync(id, ct);
    var usage = new TokenUsage();
    foreach (var turn in turns)
        if (turn.Usage is { } reported) usage += reported;
    return Results.Ok(new
    {
        usage,
        tokenCostUsd = hasRates ? TokenCostCalculator.Estimate(usage, rates).ProviderCost : (decimal?)null,
        reportedTurns = turns.Count(turn => turn.Usage is not null),
        totalTurns = turns.Count,
        complete = turns.All(turn => turn.Status is "completed" or "failed" or "cancelled" && turn.Usage is not null)
    });
});
app.MapPost("/api/agents/{id}/cancel", async (string id, IAiInferenceClient client, CancellationToken ct) =>
{
    await client.CancelHostedAgentTurnAsync(id, ct);
    return Results.NoContent();
});
app.MapDelete("/api/agents/{id}", async (string id, IAiInferenceClient client, CancellationToken ct) =>
{
    await client.DeleteHostedAgentSessionAsync(id, ct);
    return Results.NoContent();
});

app.MapPost("/api/agents/{id}/turn", async (string id, TurnRequest request, IAiInferenceClient client, HttpContext context) =>
{
    if (string.IsNullOrWhiteSpace(request.Message))
    {
        context.Response.StatusCode = 400;
        return;
    }
    context.Response.ContentType = "application/x-ndjson; charset=utf-8";
    try
    {
        await foreach (var item in client.RunHostedAgentTurnAsync(id, request.Message, context.RequestAborted))
        {
            await context.Response.WriteAsync(JsonSerializer.Serialize(item, JsonSerializerOptions.Web) + "\n", context.RequestAborted);
            await context.Response.Body.FlushAsync(context.RequestAborted);
        }
    }
    catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested) { }
    catch (Exception ex)
    {
        if (!context.RequestAborted.IsCancellationRequested)
        {
            await context.Response.WriteAsync(JsonSerializer.Serialize(new { type = "playground.error", error = ex.Message }, JsonSerializerOptions.Web) + "\n");
            await context.Response.Body.FlushAsync();
        }
    }
});

app.Run();

internal sealed record ChatRequest(string Model, string? SystemMessage, IReadOnlyList<ChatEntry> Messages);
internal sealed record ChatEntry(string Role, string Content);
internal sealed record CreateAgentRequest(string Model, string? Instructions, string? ReasoningEffort,
    bool EnableWebSearch, bool AllowNetworkAccess, string? McpLabel, string? McpUrl, string? McpAllowedTools);
internal sealed record TurnRequest(string Message);

internal static class AgentRates
{
    public static bool TryGetValue(string model, out TokenCostRates rates)
    {
        var descriptor = AiModelCatalog.GetByProvider(ProviderType.OpenAi)
            .FirstOrDefault(item => string.Equals(item.ApiName, model, StringComparison.OrdinalIgnoreCase));
        if (descriptor is not null)
            return AiModelCostCatalog.TryGetRates(descriptor.Model, out rates);
        rates = null!;
        return false;
    }
}
