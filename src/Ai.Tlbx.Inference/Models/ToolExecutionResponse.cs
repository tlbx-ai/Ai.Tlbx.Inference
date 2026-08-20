namespace Ai.Tlbx.Inference;

public sealed record ToolExecutionResponse<T>
{
    public required T Content { get; init; }
    public required TokenUsage Usage { get; init; }
    public required int Iterations { get; init; }
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public CompletionDiagnostics? Diagnostics { get; init; }
    public GroundingResult? Grounding { get; init; }
}
