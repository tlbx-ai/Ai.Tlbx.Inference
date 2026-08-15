namespace Ai.Tlbx.Inference;

public readonly record struct TokenUsage
{
    public int InputTokens { get; init; }
    /// <summary>
    /// Total provider-billed output tokens, including reasoning tokens when the provider bills them as output.
    /// </summary>
    public int OutputTokens { get; init; }
    public int CacheReadTokens { get; init; }
    public int CacheWriteTokens { get; init; }
    /// <summary>
    /// Diagnostic subset of <see cref="OutputTokens"/> spent on reasoning.
    /// </summary>
    public int ThinkingTokens { get; init; }
    public int TotalTokens => InputTokens + OutputTokens;

    public static TokenUsage operator +(TokenUsage a, TokenUsage b) => new()
    {
        InputTokens = a.InputTokens + b.InputTokens,
        OutputTokens = a.OutputTokens + b.OutputTokens,
        CacheReadTokens = a.CacheReadTokens + b.CacheReadTokens,
        CacheWriteTokens = a.CacheWriteTokens + b.CacheWriteTokens,
        ThinkingTokens = a.ThinkingTokens + b.ThinkingTokens,
    };
}
