namespace Ai.Tlbx.Inference;

public sealed record EmbeddingRequest
{
    public required EmbeddingModel Model { get; init; }
    public required string Input { get; init; }
    public int? Dimensions { get; init; }
}
