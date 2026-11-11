namespace Ai.Tlbx.Inference;

/// <summary>Native compaction only; never substitutes application-written summaries.</summary>
public sealed record CompactionPolicy
{
    public int? ThresholdTokens { get; init; }
    public int SafetyMarginTokens { get; init; } = 32_768;
}

public sealed record CompactionRequest
{
    public required AiModel Model { get; init; }
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public string? SystemMessage { get; init; }
    public string? ServiceTier { get; init; }
    public string? PromptCacheKey { get; init; }
    public int? InputTokens { get; init; }
}

public sealed record CompactionResponse
{
    public required string Id { get; init; }
    /// <summary>The entire native output window, in order. Retain every item unchanged.</summary>
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    public required TokenUsage Usage { get; init; }
    public string? ServiceTier { get; init; }
}

public sealed record CompactionState
{
    public required IReadOnlyList<ChatMessage> Messages { get; init; }
    /// <summary>Number of messages covered in the caller's full, append-only transcript.</summary>
    public required int SourceMessageCount { get; init; }
    public string? SystemMessage { get; init; }
}

public sealed record ConversationCompaction
{
    public required CompactionResponse Response { get; init; }
    public required CompactionState State { get; init; }
    public required IReadOnlyList<ChatMessage> SourceMessages { get; init; }
    public required int InputTokens { get; init; }
    public required int CompactedInputTokens { get; init; }
}

public interface ICompactionObserver
{
    ValueTask OnStartingAsync(CompactionRequest request, CancellationToken cancellationToken);
    ValueTask OnCompletedAsync(ConversationCompaction result, CancellationToken cancellationToken);
    ValueTask OnFailedAsync(Exception error, CancellationToken cancellationToken);
}

/// <summary>Receives complete newly generated journal messages, before tool execution.</summary>
/// <remarks>Callbacks must bound their own persistence work. Completed output is delivered even when the request was canceled.</remarks>
public interface IConversationObserver
{
    ValueTask OnMessageAsync(ChatMessage message, CancellationToken cancellationToken);
}
