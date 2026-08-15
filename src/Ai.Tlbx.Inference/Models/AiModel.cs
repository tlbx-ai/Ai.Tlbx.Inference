namespace Ai.Tlbx.Inference;

public enum AiModel
{
    // OpenAI
    Gpt52,
    Gpt52Pro,
    Gpt52Chat,
    Gpt53Chat,
    Gpt54,
    Gpt56Luna,

    // Anthropic
    ClaudeOpus46,
    ClaudeSonnet46,
    ClaudeHaiku45,

    // Google
    Gemini35Flash,
    Gemini3FlashPreview,
    Gemini31ProPreview,
    Gemini31FlashLitePreview,

    // xAI
    Grok41Fast,
    Grok41FastNonReasoning,
    Grok4
}

public static class AiModelExtensions
{
    public static string ToApiName(this AiModel model) => AiModelCatalog.Get(model).ApiName;

    public static string ToDisplayName(this AiModel model) => AiModelCatalog.Get(model).DisplayName;

    public static ProviderType GetProvider(this AiModel model) => AiModelCatalog.Get(model).Provider;

    public static bool SupportsThinking(this AiModel model) => AiModelCatalog.Get(model).Capabilities.SupportsThinking;

    public static int GetContextWindow(this AiModel model) => AiModelCatalog.Get(model).ContextWindow;
}
