namespace Ai.Tlbx.Inference;

public sealed record AiModelValidationResult
{
    public required AiModel Model { get; init; }
    public required bool IsValid { get; init; }
    public required IReadOnlyList<string> Errors { get; init; }
    public required IReadOnlyList<string> Warnings { get; init; }
}
