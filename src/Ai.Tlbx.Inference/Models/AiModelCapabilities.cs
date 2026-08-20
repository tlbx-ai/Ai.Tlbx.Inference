namespace Ai.Tlbx.Inference;

public sealed record AiModelCapabilities
{
    public required bool SupportsThinking { get; init; }
    public required bool SupportsTools { get; init; }
    public required bool SupportsStructuredOutput { get; init; }
    public required bool SupportsStreaming { get; init; }
    public required bool SupportsDocumentAttachments { get; init; }
    public required bool SupportsChatCompletionsApi { get; init; }
    public required bool SupportsResponsesApi { get; init; }
    public required bool SupportsWebGrounding { get; init; }
    public required bool SupportsImageSearch { get; init; }
    public required bool SupportsXSearch { get; init; }
    public required bool IsPreview { get; init; }
    public required bool RequiresReasoningBudgetHeadroom { get; init; }
    public required int DefaultSmokeMaxTokens { get; init; }
    public required int RetrySmokeMaxTokens { get; init; }
    public required int StreamingSmokeMaxTokens { get; init; }
    public required int AttachmentSmokeMaxTokens { get; init; }
}
