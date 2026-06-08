namespace Ai.Tlbx.Inference;

public sealed record TokenCostRates
{
    public required string ProviderId { get; init; }
    public required string ModelApiName { get; init; }
    public string Currency { get; init; } = "USD";
    public required decimal InputPerMillion { get; init; }
    public decimal? CachedInputPerMillion { get; init; }
    public decimal? CacheWriteInputPerMillion { get; init; }
    public required decimal OutputPerMillion { get; init; }
}

public sealed record TokenCostEstimate
{
    public required string ProviderId { get; init; }
    public required string ModelApiName { get; init; }
    public required string Currency { get; init; }
    public required int InputTokens { get; init; }
    public required int UncachedInputTokens { get; init; }
    public required int CacheReadTokens { get; init; }
    public required int CacheWriteTokens { get; init; }
    public required int OutputTokens { get; init; }
    public required decimal ProviderCost { get; init; }
    public required decimal CustomerCost { get; init; }
    public required decimal MarkupPercent { get; init; }
}

public static class TokenCostCalculator
{
    public static TokenCostEstimate Estimate(
        TokenUsage usage,
        TokenCostRates rates,
        decimal markupPercent = 0m)
    {
        ArgumentNullException.ThrowIfNull(rates);

        if (markupPercent < 0m)
        {
            throw new ArgumentOutOfRangeException(nameof(markupPercent), "Markup percent must not be negative.");
        }

        var cacheReadTokens = Math.Max(0, usage.CacheReadTokens);
        var cacheWriteTokens = Math.Max(0, usage.CacheWriteTokens);
        var uncachedInputTokens = Math.Max(0, usage.InputTokens - cacheReadTokens - cacheWriteTokens);
        var outputTokens = Math.Max(0, usage.OutputTokens + usage.ThinkingTokens);

        var providerCost =
            CostForTokens(uncachedInputTokens, rates.InputPerMillion) +
            CostForTokens(cacheReadTokens, rates.CachedInputPerMillion ?? rates.InputPerMillion) +
            CostForTokens(cacheWriteTokens, rates.CacheWriteInputPerMillion ?? rates.InputPerMillion) +
            CostForTokens(outputTokens, rates.OutputPerMillion);

        var customerCost = providerCost * (1m + markupPercent / 100m);

        return new TokenCostEstimate
        {
            ProviderId = rates.ProviderId,
            ModelApiName = rates.ModelApiName,
            Currency = rates.Currency,
            InputTokens = Math.Max(0, usage.InputTokens),
            UncachedInputTokens = uncachedInputTokens,
            CacheReadTokens = cacheReadTokens,
            CacheWriteTokens = cacheWriteTokens,
            OutputTokens = outputTokens,
            ProviderCost = providerCost,
            CustomerCost = customerCost,
            MarkupPercent = markupPercent
        };
    }

    private static decimal CostForTokens(int tokens, decimal pricePerMillion)
        => tokens <= 0 ? 0m : tokens / 1_000_000m * pricePerMillion;
}

public static class AiModelCostCatalog
{
    private static readonly IReadOnlyDictionary<AiModel, TokenCostRates> _aiModelRates =
        new Dictionary<AiModel, TokenCostRates>
        {
            [AiModel.Gpt52] = OpenAi("gpt-5.2", input: 1.75m, cachedInput: 0.175m, output: 14m),
            [AiModel.Gpt52Pro] = OpenAi("gpt-5.2-pro", input: 15m, cachedInput: null, output: 90m),
            [AiModel.Gpt52Chat] = OpenAi("gpt-5.2-chat-latest", input: 1.75m, cachedInput: 0.175m, output: 14m),
            [AiModel.Gpt53Chat] = OpenAi("gpt-5.3-chat-latest", input: 1.75m, cachedInput: 0.175m, output: 14m),
            [AiModel.Gpt54] = OpenAi("gpt-5.4", input: 1.25m, cachedInput: 0.13m, output: 7.5m),
            [AiModel.ClaudeOpus46] = Anthropic("claude-opus-4-6", input: 5m, cacheWrite: 6.25m, cacheRead: 0.5m, output: 25m),
            [AiModel.ClaudeSonnet46] = Anthropic("claude-sonnet-4-6", input: 3m, cacheWrite: 3.75m, cacheRead: 0.3m, output: 15m),
            [AiModel.ClaudeHaiku45] = Anthropic("claude-haiku-4-5-20251001", input: 1m, cacheWrite: 1.25m, cacheRead: 0.1m, output: 5m),
            [AiModel.Gemini35Flash] = Google("gemini-3.5-flash", input: 1.5m, cachedInput: 0.15m, output: 9m),
            [AiModel.Gemini3FlashPreview] = Google("gemini-3-flash-preview", input: 0.5m, cachedInput: 0.05m, output: 3m),
            [AiModel.Gemini31ProPreview] = Google("gemini-3.1-pro-preview", input: 2m, cachedInput: 0.2m, output: 12m),
            [AiModel.Gemini31FlashLitePreview] = Google("gemini-3.1-flash-lite-preview", input: 0.25m, cachedInput: 0.025m, output: 1.5m),
            [AiModel.Grok41Fast] = Xai("grok-4-1-fast-reasoning", input: 1.25m, cachedInput: 0.2m, output: 2.5m),
            [AiModel.Grok41FastNonReasoning] = Xai("grok-4-1-fast", input: 1.25m, cachedInput: 0.2m, output: 2.5m),
            [AiModel.Grok4] = Xai("grok-4", input: 1.25m, cachedInput: 0.2m, output: 2.5m)
        };

    private static readonly IReadOnlyDictionary<EmbeddingModel, TokenCostRates> _embeddingModelRates =
        new Dictionary<EmbeddingModel, TokenCostRates>
        {
            [EmbeddingModel.TextEmbedding3Large] = OpenAi("text-embedding-3-large", input: 0.13m, cachedInput: null, output: 0m),
            [EmbeddingModel.TextEmbedding3Small] = OpenAi("text-embedding-3-small", input: 0.02m, cachedInput: null, output: 0m),
            [EmbeddingModel.GeminiEmbedding001] = Google("gemini-embedding-001", input: 0.15m, cachedInput: null, output: 0m)
        };

    public static bool TryGetRates(AiModel model, out TokenCostRates rates)
        => _aiModelRates.TryGetValue(model, out rates!);

    public static bool TryGetRates(EmbeddingModel model, out TokenCostRates rates)
        => _embeddingModelRates.TryGetValue(model, out rates!);

    public static TokenCostRates GetRates(AiModel model)
        => TryGetRates(model, out var rates)
            ? rates
            : throw new ArgumentOutOfRangeException(nameof(model), model, "No token cost rates are registered for this model.");

    public static TokenCostRates GetRates(EmbeddingModel model)
        => TryGetRates(model, out var rates)
            ? rates
            : throw new ArgumentOutOfRangeException(nameof(model), model, "No token cost rates are registered for this model.");

    public static TokenCostEstimate EstimateCost(this TokenUsage usage, AiModel model, decimal markupPercent = 0m)
        => TokenCostCalculator.Estimate(usage, GetRates(model), markupPercent);

    public static TokenCostEstimate EstimateCost(this TokenUsage usage, EmbeddingModel model, decimal markupPercent = 0m)
        => TokenCostCalculator.Estimate(usage, GetRates(model), markupPercent);

    private static TokenCostRates OpenAi(string model, decimal input, decimal? cachedInput, decimal output)
        => Create("openai", model, input, cachedInput, cacheWrite: null, output);

    private static TokenCostRates Anthropic(string model, decimal input, decimal cacheWrite, decimal cacheRead, decimal output)
        => Create("anthropic", model, input, cacheRead, cacheWrite, output);

    private static TokenCostRates Google(string model, decimal input, decimal? cachedInput, decimal output)
        => Create("google", model, input, cachedInput, cacheWrite: null, output);

    private static TokenCostRates Xai(string model, decimal input, decimal? cachedInput, decimal output)
        => Create("xai", model, input, cachedInput, cacheWrite: null, output);

    private static TokenCostRates Create(
        string providerId,
        string model,
        decimal input,
        decimal? cachedInput,
        decimal? cacheWrite,
        decimal output)
        => new()
        {
            ProviderId = providerId,
            ModelApiName = model,
            InputPerMillion = input,
            CachedInputPerMillion = cachedInput,
            CacheWriteInputPerMillion = cacheWrite,
            OutputPerMillion = output
        };
}
