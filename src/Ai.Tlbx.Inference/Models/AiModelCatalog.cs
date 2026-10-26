namespace Ai.Tlbx.Inference;

public static class AiModelCatalog
{
    private static readonly IReadOnlyDictionary<AiModel, AiModelDescriptor> _models =
        new Dictionary<AiModel, AiModelDescriptor>
        {
            [AiModel.Gpt52] = Create(
                AiModel.Gpt52, "gpt-5.2", "GPT-5.2", ProviderType.OpenAi, 400000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: false, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Gpt52Pro] = Create(
                AiModel.Gpt52Pro, "gpt-5.2-pro", "GPT-5.2 Pro", ProviderType.OpenAi, 400000,
                ModelEndpointFamily.Responses,
                supportsThinking: false, supportsChatCompletionsApi: false, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256,
                notes: "Use the Responses API for normal completions."),
            [AiModel.Gpt52Chat] = Create(
                AiModel.Gpt52Chat, "gpt-5.2-chat-latest", "GPT-5.2 Chat", ProviderType.OpenAi, 128000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: false, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Gpt53Chat] = Create(
                AiModel.Gpt53Chat, "gpt-5.3-chat-latest", "GPT-5.3 Chat", ProviderType.OpenAi, 128000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: false, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Gpt54] = Create(
                AiModel.Gpt54, "gpt-5.4", "GPT-5.4", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: false, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Gpt56Luna] = Create(
                AiModel.Gpt56Luna, "gpt-5.6-luna", "GPT-5.6 Luna", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.Responses,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gpt6Luna] = Create(
                AiModel.Gpt6Luna, "gpt-6-luna", "GPT-6 Luna", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.Responses,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 2048, retrySmokeMaxTokens: 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gpt6Sol] = Create(
                AiModel.Gpt6Sol, "gpt-6-sol", "GPT-6 Sol", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.Responses,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 2048, retrySmokeMaxTokens: 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gpt61Sol] = Create(
                AiModel.Gpt61Sol, "gpt-6.1-sol", "GPT-6.1 Sol", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.Responses,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: true,
                defaultSmokeMaxTokens: 2048, retrySmokeMaxTokens: 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.ClaudeOpus46] = Create(
                AiModel.ClaudeOpus46, "claude-opus-4-6", "Claude Opus 4.6", ProviderType.Anthropic, 200000,
                ModelEndpointFamily.AnthropicMessages,
                supportsThinking: true, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.ClaudeSonnet46] = Create(
                AiModel.ClaudeSonnet46, "claude-sonnet-4-6", "Claude Sonnet 4.6", ProviderType.Anthropic, 200000,
                ModelEndpointFamily.AnthropicMessages,
                supportsThinking: true, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.ClaudeHaiku45] = Create(
                AiModel.ClaudeHaiku45, "claude-haiku-4-5-20251001", "Claude Haiku 4.5", ProviderType.Anthropic, 200000,
                ModelEndpointFamily.AnthropicMessages,
                supportsThinking: false, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Gemini35Flash] = Create(
                AiModel.Gemini35Flash, "gemini-3.5-flash", "Gemini 3.5 Flash", ProviderType.Google, 1048576,
                ModelEndpointFamily.GoogleGenerateContent,
                supportsThinking: true, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 512, retrySmokeMaxTokens: 1024,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gemini3FlashPreview] = Create(
                AiModel.Gemini3FlashPreview, "gemini-3-flash-preview", "Gemini 3 Flash Preview", ProviderType.Google, 1000000,
                ModelEndpointFamily.GoogleGenerateContent,
                supportsThinking: true, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                isPreview: true, defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gemini31ProPreview] = Create(
                AiModel.Gemini31ProPreview, "gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", ProviderType.Google, 1000000,
                ModelEndpointFamily.GoogleGenerateContent,
                supportsThinking: true, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                isPreview: true, defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gemini31FlashLitePreview] = Create(
                AiModel.Gemini31FlashLitePreview, "gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash-Lite Preview", ProviderType.Google, 1000000,
                ModelEndpointFamily.GoogleGenerateContent,
                supportsThinking: false, supportsChatCompletionsApi: false, supportsResponsesApi: false,
                isPreview: true, defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Grok41Fast] = Create(
                AiModel.Grok41Fast, "grok-4-1-fast-reasoning", "Grok 4.1 Fast", ProviderType.Xai, 1000000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256,
                notes: "Retired 2026-05-15; provider redirects this identifier to Grok 4.3 at its current prices. Prefer Grok43 or Grok47."),
            [AiModel.Grok41FastNonReasoning] = Create(
                AiModel.Grok41FastNonReasoning, "grok-4-1-fast-non-reasoning", "Grok 4.1 Fast (Non-Reasoning)", ProviderType.Xai, 1000000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: false, supportsChatCompletionsApi: true, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256,
                notes: "Retired 2026-05-15; provider redirects this identifier to Grok 4.3 with reasoning disabled. Prefer Grok43 with ReasoningEffort=none."),
            [AiModel.Grok4] = Create(
                AiModel.Grok4, "grok-4", "Grok 4", ProviderType.Xai, 256000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            // Official provider catalog review: 2026-10-27. Append enum values in AiModel.
            [AiModel.Gpt6Astra] = Create(
                AiModel.Gpt6Astra, "gpt-6-astra", "GPT-6 Astra", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.Responses, true, true, true, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gpt61Terra] = Create(
                AiModel.Gpt61Terra, "gpt-6.1-terra", "GPT-6.1 Terra", ProviderType.OpenAi, 1050000,
                ModelEndpointFamily.Responses, true, true, true, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.ChatLatest] = Create(
                AiModel.ChatLatest, "chat-latest", "Chat Latest", ProviderType.OpenAi, 400000,
                ModelEndpointFamily.Responses, false, true, true, 256, 512,
                notes: "Rolling alias for the current ChatGPT Instant model; behavior may change."),
            [AiModel.ClaudeFable51] = Create(
                AiModel.ClaudeFable51, "claude-fable-5-1", "Claude Fable 5.1", ProviderType.Anthropic, 1000000,
                ModelEndpointFamily.AnthropicMessages, true, false, false, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.ClaudeOpus55] = Create(
                AiModel.ClaudeOpus55, "claude-opus-5-5", "Claude Opus 5.5", ProviderType.Anthropic, 1000000,
                ModelEndpointFamily.AnthropicMessages, true, false, false, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.ClaudeSonnet55] = Create(
                AiModel.ClaudeSonnet55, "claude-sonnet-5-5", "Claude Sonnet 5.5", ProviderType.Anthropic, 1000000,
                ModelEndpointFamily.AnthropicMessages, true, false, false, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.ClaudeHaiku55] = Create(
                AiModel.ClaudeHaiku55, "claude-haiku-5-5", "Claude Haiku 5.5", ProviderType.Anthropic, 1000000,
                ModelEndpointFamily.AnthropicMessages, true, false, false, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gemini4Argon] = Create(
                AiModel.Gemini4Argon, "gemini-4-argon", "Gemini 4 Argon", ProviderType.Google, 1048576,
                ModelEndpointFamily.GoogleGenerateContent, true, false, false, 2048, 4096,
                requiresReasoningBudgetHeadroom: true,
                notes: "Unified input/output window. REST output cap 32,768; streaming cap 65,536. Long decodes need explicit continuation."),
            [AiModel.Gemini38Flash] = Create(
                AiModel.Gemini38Flash, "gemini-3.8-flash", "Gemini 3.8 Flash", ProviderType.Google, 1048576,
                ModelEndpointFamily.GoogleGenerateContent, true, false, false, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gemini35FlashLite] = Create(
                AiModel.Gemini35FlashLite, "gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", ProviderType.Google, 1048576,
                ModelEndpointFamily.GoogleGenerateContent, true, false, false, 1024, 2048,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Gemini31FlashLite] = Create(
                AiModel.Gemini31FlashLite, "gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", ProviderType.Google, 1048576,
                ModelEndpointFamily.GoogleGenerateContent, true, false, false, 1024, 2048,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Grok47] = Create(
                AiModel.Grok47, "grok-4.7", "Grok 4.7", ProviderType.Xai, 500000,
                ModelEndpointFamily.Responses, true, true, true, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Grok46] = Create(
                AiModel.Grok46, "grok-4.6", "Grok 4.6", ProviderType.Xai, 500000,
                ModelEndpointFamily.Responses, true, true, true, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
            [AiModel.Grok43] = Create(
                AiModel.Grok43, "grok-4.3", "Grok 4.3", ProviderType.Xai, 1000000,
                ModelEndpointFamily.Responses, true, true, true, 2048, 4096,
                requiresReasoningBudgetHeadroom: true),
        };

    public static IReadOnlyList<AiModelDescriptor> All { get; } = _models.Values.OrderBy(m => (int)m.Provider).ThenBy(m => m.DisplayName).ToArray();

    public static AiModelDescriptor Get(AiModel model)
    {
        if (_models.TryGetValue(model, out var descriptor))
        {
            return descriptor;
        }

        throw new ArgumentOutOfRangeException(nameof(model), model, null);
    }

    public static IReadOnlyList<AiModelDescriptor> GetByProvider(ProviderType providerType)
        => All.Where(model => model.Provider == providerType).ToArray();

    private static AiModelDescriptor Create(
        AiModel model,
        string apiName,
        string displayName,
        ProviderType provider,
        int contextWindow,
        ModelEndpointFamily preferredEndpoint,
        bool supportsThinking,
        bool supportsChatCompletionsApi,
        bool supportsResponsesApi,
        int defaultSmokeMaxTokens,
        int retrySmokeMaxTokens,
        bool supportsDocumentAttachments = true,
        bool isPreview = false,
        bool requiresReasoningBudgetHeadroom = false,
        int? streamingSmokeMaxTokens = null,
        int? attachmentSmokeMaxTokens = null,
        string? notes = null)
    {
        var resolvedStreamingSmokeMaxTokens = streamingSmokeMaxTokens
            ?? (requiresReasoningBudgetHeadroom ? 1024 : 768);
        var resolvedAttachmentSmokeMaxTokens = attachmentSmokeMaxTokens
            ?? (requiresReasoningBudgetHeadroom ? 768 : 512);

        return new AiModelDescriptor
        {
            Model = model,
            ApiName = apiName,
            DisplayName = displayName,
            Provider = provider,
            ContextWindow = contextWindow,
            PreferredEndpoint = preferredEndpoint,
            Notes = notes,
            Capabilities = new AiModelCapabilities
            {
                SupportsThinking = supportsThinking,
                SupportsTools = true,
                SupportsStructuredOutput = true,
                SupportsStreaming = true,
                SupportsDocumentAttachments = supportsDocumentAttachments,
                SupportsChatCompletionsApi = supportsChatCompletionsApi,
                SupportsResponsesApi = supportsResponsesApi,
                SupportsWebGrounding = true,
                SupportsImageSearch = provider is ProviderType.OpenAi or ProviderType.Xai,
                SupportsXSearch = provider == ProviderType.Xai,
                IsPreview = isPreview,
                RequiresReasoningBudgetHeadroom = requiresReasoningBudgetHeadroom,
                DefaultSmokeMaxTokens = defaultSmokeMaxTokens,
                RetrySmokeMaxTokens = retrySmokeMaxTokens,
                StreamingSmokeMaxTokens = resolvedStreamingSmokeMaxTokens,
                AttachmentSmokeMaxTokens = resolvedAttachmentSmokeMaxTokens,
            },
        };
    }
}
