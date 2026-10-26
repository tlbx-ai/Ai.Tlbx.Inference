using System.Text;
using System.Text.Json;

namespace Ai.Tlbx.Inference.IntegrationTests.Providers;

// Manual, bounded provider acceptance. Each call uses default tier, low effort,
// at most 2048 output tokens and a three-iteration tool cap. Never run in CI.
public sealed class CatalogMaintenanceTests
{
    private static readonly SpendingObserver Budget = new();

    [Theory]
    [InlineData(AiModel.Gpt6Astra)]
    [InlineData(AiModel.Gpt61Terra)]
    [InlineData(AiModel.ChatLatest)]
    [InlineData(AiModel.ClaudeFable51)]
    [InlineData(AiModel.ClaudeOpus55)]
    [InlineData(AiModel.ClaudeSonnet55)]
    [InlineData(AiModel.ClaudeHaiku55)]
    [InlineData(AiModel.Gemini4Argon)]
    [InlineData(AiModel.Gemini38Flash)]
    [InlineData(AiModel.Gemini35FlashLite)]
    [InlineData(AiModel.Gemini31FlashLite)]
    [InlineData(AiModel.Grok47)]
    [InlineData(AiModel.Grok46)]
    [InlineData(AiModel.Grok43)]
    public async Task ToolLoop_ReturnsValueKnownOnlyToTool(AiModel model)
    {
        using var context = IntegrationScenarioHelper.CreateClient(model.GetProvider());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(90));
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{},"required":[],"additionalProperties":false}""");
        var executions = 0;
        var response = await context.Client.CompleteWithToolsAsync(Request(model), [Tool(schema)], call =>
        {
            executions++;
            return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "pine-7426" });
        }, maxIterations: 3, ct: timeout.Token);
        Assert.InRange(executions, 1, 2);
        Assert.Contains("pine-7426", response.Content);
        Assert.True(response.Usage.TotalTokens > 0);
    }

    [Theory]
    [InlineData(AiModel.Gpt61Terra)]
    [InlineData(AiModel.ClaudeHaiku55)]
    [InlineData(AiModel.Gemini38Flash)]
    [InlineData(AiModel.Grok47)]
    public async Task StreamingToolLoop_PreservesProviderContinuation(AiModel model)
    {
        using var context = IntegrationScenarioHelper.CreateClient(model.GetProvider());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(90));
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{},"required":[],"additionalProperties":false}""");
        var text = new StringBuilder();
        var executions = 0;
        var completed = false;
        await foreach (var item in context.Client.StreamWithToolsAsync(Request(model), [Tool(schema)], call =>
        {
            executions++;
            return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "pine-7426" });
        }, maxIterations: 3, ct: timeout.Token))
        {
            if (item is TextDeltaEvent delta) text.Append(delta.Text);
            if (item is CompletedEvent) completed = true;
        }
        Assert.True(completed);
        Assert.InRange(executions, 1, 2);
        Assert.Contains("pine-7426", text.ToString());
    }

    [Theory]
    [InlineData(AiModel.Gpt6Astra)]
    [InlineData(AiModel.Gpt61Terra)]
    [InlineData(AiModel.ChatLatest)]
    [InlineData(AiModel.ClaudeFable51)]
    [InlineData(AiModel.ClaudeHaiku55)]
    [InlineData(AiModel.Gemini4Argon)]
    [InlineData(AiModel.Gemini38Flash)]
    [InlineData(AiModel.Grok47)]
    public async Task StructuredOutput_ReturnsRequestedJson(AiModel model)
    {
        using var context = IntegrationScenarioHelper.CreateClient(model.GetProvider());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(90));
        Budget.ReserveFlat(0.15m);
        var response = await context.Client.CompleteAsync(Request(model) with
        {
            ToolIterationObserver = null,
            SystemMessage = null,
            JsonSchema = """{"type":"object","properties":{"code":{"type":"integer"}},"required":["code"],"additionalProperties":false}""",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Return code 42 in the requested JSON object." }]
        }, timeout.Token);
        using var parsed = JsonDocument.Parse(response.Content);
        Assert.Equal(42, parsed.RootElement.GetProperty("code").GetInt32());
        Assert.True(response.Usage.TotalTokens > 0);
    }

    [Theory]
    [InlineData(ImageGenerationModel.GptImage25Flare, true)]
    [InlineData(ImageGenerationModel.GptImage25Sunburst, false)]
    [InlineData(ImageGenerationModel.GeminiNanoBanana21, true)]
    [InlineData(ImageGenerationModel.Gemini31FlashImage, false)]
    [InlineData(ImageGenerationModel.Gemini3ProImage, false)]
    public async Task ImageGenerationAndEditing_ReturnImageBytes(ImageGenerationModel model, bool edit)
    {
        using var context = IntegrationScenarioHelper.CreateClient(model.GetProvider());
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        Budget.ReserveFlat(edit ? 0.40m : 0.20m);
        var bytes = await context.Client.GenerateImageAsync(new ImageGenerationRequest
        {
            Model = model, Prompt = "Create an original abstract geometric study on white: one irregular teal hexagon at the left, three small amber triangles arranged diagonally at the right, and a thin violet arc below. Flat vector style, no text, no logos.",
            Quality = model.GetProvider() == ProviderType.OpenAi ? "low" : null,
            Size = model.GetProvider() == ProviderType.Google ? "1K" : "1024x1024"
        }, timeout.Token);
        Assert.True(bytes.Length > 1000);
        var directory = Environment.GetEnvironmentVariable("INFERENCE_MAINTENANCE_ARTIFACTS")
            ?? Path.Combine(AppContext.BaseDirectory, "TestResults");
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(Path.Combine(directory, $"{model}-generated.png"), bytes, timeout.Token);
        if (!edit) return;
        var image = new ImageEditInput { Content = bytes, MimeType = "image/png", FileName = "circle.png" };
        var edited = await context.Client.EditImageAsync(new ImageEditRequest
        {
            Model = model, Prompt = "Change the blue circle to red. Keep the white background.", Images = [image, image],
            Quality = model.GetProvider() == ProviderType.OpenAi ? "low" : null,
            Size = model.GetProvider() == ProviderType.Google ? "1K" : "1024x1024"
        }, timeout.Token);
        Assert.True(edited.Length > 1000);
        await File.WriteAllBytesAsync(Path.Combine(directory, $"{model}-edited.png"), edited, timeout.Token);
    }

    [Fact]
    public async Task Embedding2_SingleAndBatchPreserveDimensionsAndUsage()
    {
        using var context = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        Budget.ReserveFlat(0.01m);
        var single = await context.Client.EmbedAsync(new EmbeddingRequest
        {
            Model = EmbeddingModel.GeminiEmbedding2, Input = "A blue circle."
        }, timeout.Token);
        Assert.Equal(3072, single.Embedding.Length);
        Assert.True(single.Usage.InputTokens > 0);
        var batch = await context.Client.EmbedBatchAsync(new BatchEmbeddingRequest
        {
            Model = EmbeddingModel.GeminiEmbedding2, Inputs = ["A blue circle.", "A red square."]
        }, timeout.Token);
        Assert.Equal(2, batch.Embeddings.Count);
        Assert.All(batch.Embeddings, item => Assert.Equal(3072, item.Length));
        Assert.True(batch.Usage.InputTokens > 0);
    }

    private static CompletionRequest Request(AiModel model) => new()
    {
        Model = model,
        ReasoningEffort = model.SupportsThinking() ? "low" : null,
        MaxTokens = 2048,
        SystemMessage = "Call read_code once. Reply only with the exact code it returns. Never invent a code.",
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Read the current code." }],
        ToolIterationObserver = Budget
    };

    private static ToolDefinition Tool(JsonDocument schema) => new()
    {
        Name = "read_code", Description = "Returns the current secret code.", ParametersSchema = schema.RootElement
    };

    private sealed class SpendingObserver : IToolIterationObserver
    {
        private decimal _spent;
        private decimal _reserved;
        private AiModel _model;
        public void ReserveFlat(decimal amount)
        {
            if (_spent + amount > 4m) throw new InvalidOperationException("Manual maintenance test USD4 budget exhausted.");
            _spent += amount;
            Console.WriteLine($"COST reservationUSD={amount:F8} totalUSD={_spent:F8}");
        }
        public ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken)
        {
            _model = request.Request.Model;
            var rates = AiModelCostCatalog.GetRates(_model);
            // Inputs are short, local fixed prompts/tool results. Reserve generously;
            // retain a reservation for missing usage so a failed call is not free.
            _reserved = (8192m * rates.InputPerMillion + 2048m * rates.OutputPerMillion) / 1_000_000m;
            if (_spent + _reserved > 4m) throw new InvalidOperationException("Manual maintenance test USD4 budget exhausted.");
            return ValueTask.CompletedTask;
        }
        public ValueTask OnFinishedAsync(ToolIterationResult result)
        {
            var cost = result.Usage is TokenUsage usage ? usage.EstimateCost(_model).ProviderCost : _reserved;
            _spent += cost;
            Console.WriteLine($"COST model={_model} iteration={result.Iteration} usd={cost:F8} totalUSD={_spent:F8} complete={result.ResponseCompleted}");
            return ValueTask.CompletedTask;
        }
    }
}
