namespace Ai.Tlbx.Inference.Providers;

internal interface IProvider
{
    Task<ProviderResponse> CompleteAsync(ProviderRequest request, CancellationToken ct);
    IAsyncEnumerable<ProviderStreamEvent> StreamAsync(ProviderRequest request, CancellationToken ct);
    Task<ProviderEmbeddingResponse> EmbedAsync(ProviderEmbeddingRequest request, CancellationToken ct);
    Task<ProviderBatchEmbeddingResponse> EmbedBatchAsync(ProviderBatchEmbeddingRequest request, CancellationToken ct);
    Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct);
}

internal sealed record ProviderRequest
{
    public required string ModelApiName { get; init; }
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public string? SystemMessage { get; init; }
    public double? Temperature { get; init; }
    public int? MaxTokens { get; init; }
    public int? ThinkingBudget { get; init; }
    public bool EnableCache { get; init; }
    public string? JsonSchema { get; init; }
    public double? TopP { get; init; }
    public IReadOnlyList<string>? StopSequences { get; init; }
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }
}

internal sealed record ProviderResponse
{
    public required string Content { get; init; }
    public required TokenUsage Usage { get; init; }
    public string? StopReason { get; init; }
    public IReadOnlyList<ToolCallRequest>? ToolCalls { get; init; }
}

internal sealed record ProviderStreamEvent
{
    public string? TextDelta { get; init; }
    public ToolCallRequest? ToolCall { get; init; }
    public TokenUsage? Usage { get; init; }
}

internal sealed record ProviderEmbeddingRequest
{
    public required string ModelApiName { get; init; }
    public required string Input { get; init; }
    public int? Dimensions { get; init; }
}

internal sealed record ProviderBatchEmbeddingRequest
{
    public required string ModelApiName { get; init; }
    public required IReadOnlyList<string> Inputs { get; init; }
    public int? Dimensions { get; init; }
}

internal sealed record ProviderEmbeddingResponse
{
    public required ReadOnlyMemory<float> Embedding { get; init; }
    public required TokenUsage Usage { get; init; }
}

internal sealed record ProviderBatchEmbeddingResponse
{
    public required IReadOnlyList<ReadOnlyMemory<float>> Embeddings { get; init; }
    public required TokenUsage Usage { get; init; }
}

internal sealed record ProviderImageRequest
{
    public required string Prompt { get; init; }
    public string? Size { get; init; }
    public string? Quality { get; init; }
}
