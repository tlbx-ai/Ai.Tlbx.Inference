namespace Ai.Tlbx.Inference;

public sealed record EmbeddingResponse
{
    public required ReadOnlyMemory<float> Embedding { get; init; }
    public required TokenUsage Usage { get; init; }
    public required EmbeddingModel Model { get; init; }
}

public sealed record BatchEmbeddingRequest
{
    public required EmbeddingModel Model { get; init; }
    public required IReadOnlyList<string> Inputs { get; init; }
    public int? Dimensions { get; init; }
}

public sealed record BatchEmbeddingResponse
{
    public required IReadOnlyList<ReadOnlyMemory<float>> Embeddings { get; init; }
    public required TokenUsage Usage { get; init; }
    public required EmbeddingModel Model { get; init; }
}
