namespace Ai.Tlbx.Inference;

public sealed record ToolExecutionResponse<T>
{
    public required T Content { get; init; }
    public string? ProviderContentJson { get; init; }
    public required TokenUsage Usage { get; init; }
    public required int Iterations { get; init; }
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public CompletionDiagnostics? Diagnostics { get; init; }
    public GroundingResult? Grounding { get; init; }
    public CompactionState? CompactionState { get; init; }
    public IReadOnlyList<ConversationCompaction> Compactions { get; init; } = [];
}
