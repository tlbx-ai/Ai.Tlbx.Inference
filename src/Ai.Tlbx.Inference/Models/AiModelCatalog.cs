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
                AiModel.Grok41Fast, "grok-4-1-fast-reasoning", "Grok 4.1 Fast", ProviderType.Xai, 2000000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Grok41FastNonReasoning] = Create(
                AiModel.Grok41FastNonReasoning, "grok-4-1-fast", "Grok 4.1 Fast (Non-Reasoning)", ProviderType.Xai, 2000000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: false, supportsChatCompletionsApi: true, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
            [AiModel.Grok4] = Create(
                AiModel.Grok4, "grok-4", "Grok 4", ProviderType.Xai, 256000,
                ModelEndpointFamily.ChatCompletions,
                supportsThinking: true, supportsChatCompletionsApi: true, supportsResponsesApi: false,
                defaultSmokeMaxTokens: 128, retrySmokeMaxTokens: 256),
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
