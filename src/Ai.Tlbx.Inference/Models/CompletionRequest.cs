namespace Ai.Tlbx.Inference;

public sealed record CompletionRequest
{
    public required AiModel Model { get; init; }
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public string? SystemMessage { get; init; }
    public double? Temperature { get; init; }
    public int? MaxTokens { get; init; }
    public int? ThinkingBudget { get; init; }
    public bool EnableCache { get; init; }
    public string? JsonSchema { get; init; }
    public double? TopP { get; init; }
    public IReadOnlyList<string>? StopSequences { get; init; }
}
