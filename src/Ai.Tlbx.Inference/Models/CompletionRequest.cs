namespace Ai.Tlbx.Inference;

public sealed record CompletionRequest
{
    public required AiModel Model { get; init; }
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public string? SystemMessage { get; init; }
    public double? Temperature { get; init; }
    public int? MaxTokens { get; init; }
    public int? ThinkingBudget { get; init; }
    public string? ReasoningEffort { get; init; }
    public string? ServiceTier { get; init; }
    public bool EnableCache { get; init; }
    public string? JsonSchema { get; init; }
    public double? TopP { get; init; }
    public IReadOnlyList<string>? StopSequences { get; init; }
    public bool EnableWebSearch { get; init; }
    public bool EnableXSearch { get; init; }
    public GroundingOptions? Grounding { get; init; }
    /// <summary>
    /// Optional per-provider-call accounting for CompleteWithToolsAsync and StreamWithToolsAsync.
    /// Provider-internal HTTP retries are part of the same call. Not used by the plain completion APIs.
    /// </summary>
    public IToolIterationObserver? ToolIterationObserver { get; init; }
}
