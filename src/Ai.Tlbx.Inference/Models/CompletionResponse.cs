namespace Ai.Tlbx.Inference;

public sealed record CompletionResponse<T>
{
    public required T Content { get; init; }
    /// <summary>Opaque provider continuation data to retain on an assistant ChatMessage when continuing the conversation.</summary>
    public string? ProviderContentJson { get; init; }
    public required TokenUsage Usage { get; init; }
    public required AiModel Model { get; init; }
    public string? StopReason { get; init; }
    public CompletionDiagnostics? Diagnostics { get; init; }
    public GroundingResult? Grounding { get; init; }
}
