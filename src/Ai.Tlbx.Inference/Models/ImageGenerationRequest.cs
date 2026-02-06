namespace Ai.Tlbx.Inference;

public sealed record ImageGenerationRequest
{
    public required string Prompt { get; init; }
    public string? Size { get; init; }
    public string? Quality { get; init; }
}
