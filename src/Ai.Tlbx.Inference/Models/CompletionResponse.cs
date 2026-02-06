namespace Ai.Tlbx.Inference;

public sealed record CompletionResponse<T>
{
    public required T Content { get; init; }
    public required TokenUsage Usage { get; init; }
    public required AiModel Model { get; init; }
    public string? StopReason { get; init; }
}
