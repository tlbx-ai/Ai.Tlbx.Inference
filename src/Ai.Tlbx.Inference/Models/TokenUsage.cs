namespace Ai.Tlbx.Inference;

public sealed record TokenUsage
{
    public int InputTokens { get; init; }
    public int OutputTokens { get; init; }
    public int CacheReadTokens { get; init; }
    public int CacheWriteTokens { get; init; }
    public int ThinkingTokens { get; init; }
    public int TotalTokens => InputTokens + OutputTokens + ThinkingTokens;

    public static TokenUsage operator +(TokenUsage a, TokenUsage b) => new()
    {
        InputTokens = a.InputTokens + b.InputTokens,
        OutputTokens = a.OutputTokens + b.OutputTokens,
        CacheReadTokens = a.CacheReadTokens + b.CacheReadTokens,
        CacheWriteTokens = a.CacheWriteTokens + b.CacheWriteTokens,
        ThinkingTokens = a.ThinkingTokens + b.ThinkingTokens,
    };
}
