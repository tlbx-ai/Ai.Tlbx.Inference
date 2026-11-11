using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ai.Tlbx.Inference.Providers;

internal abstract partial class OpenAiCompatibleProvider
{
    public async Task<int> CountInputTokensAsync(ProviderRequest request, CancellationToken ct)
    {
        // Inline files remain valid after a compaction checkpoint is persisted.
        var full = BuildResponsesRequestBody(request, false, null);
        var body = new JsonObject();
        foreach (var key in new[] { "model", "input", "tools", "reasoning", "text", "parallel_tool_calls", "tool_choice" })
            if (full[key] is { } value) body[key] = value.DeepClone();
        using var result = await SendContextRequestAsync(body, "/v1/responses/input_tokens", ct).ConfigureAwait(false);
        return result.RootElement.GetProperty("input_tokens").GetInt32();
    }

    public async Task<CompactionResponse> CompactAsync(ProviderRequest request, CancellationToken ct)
    {
        var full = BuildResponsesRequestBody(request, false, null);
        // Instructions are input developer items; never send the deprecated instructions root field.
        // Tools/reasoning are not accepted as compact endpoint root parameters.
        var body = new JsonObject { ["model"] = request.ModelApiName, ["input"] = full["input"]!.DeepClone() };
        if (request.ServiceTier is not null) body["service_tier"] = request.ServiceTier;
        if (request.PromptCacheKey is not null) body["prompt_cache_key"] = request.PromptCacheKey;
        using var result = await SendContextRequestAsync(body, "/v1/responses/compact", ct).ConfigureAwait(false);
        var root = result.RootElement;
        var output = root.GetProperty("output");
        if (output.ValueKind != JsonValueKind.Array || output.GetArrayLength() == 0)
            throw new InvalidOperationException("Native compaction returned no canonical window.");
        var messages = output.EnumerateArray().Select(item => new ChatMessage
        {
            Role = ChatRole.Provider, ProviderContentJson = item.GetRawText()
        }).ToArray();
        if (!output.EnumerateArray().Any(item => item.TryGetProperty("type", out var type) && type.GetString() == "compaction"))
            throw new InvalidOperationException("Native compaction returned no compaction item.");
        var tier = ReadServiceTier(root);
        return new CompactionResponse
        {
            Id = root.GetProperty("id").GetString()!, Messages = messages,
            Usage = ParseResponsesUsage(root.GetProperty("usage")) with { ServiceTier = tier }, ServiceTier = tier
        };
    }

    private async Task<JsonDocument> SendContextRequestAsync(JsonObject body, string path, CancellationToken ct)
    {
        var bytes = SerializeToUtf8Bytes(body);
        using var response = await _context.SendAsync(() => CreateHttpRequest(bytes, path), ct).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"OpenAI context operation {path} failed with HTTP {(int)response.StatusCode}.", null, response.StatusCode);
        await using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        return await JsonDocument.ParseAsync(stream, cancellationToken: ct).ConfigureAwait(false);
    }
}
