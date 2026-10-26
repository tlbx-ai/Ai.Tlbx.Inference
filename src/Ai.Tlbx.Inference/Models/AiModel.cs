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
    Grok4,

    // Keep existing enum values stable for persisted model selections.
    Gpt6Luna,
    Gpt6Sol,
    Gpt61Sol,

    // Append new entries: numeric values are persisted by consumers.
    Gpt6Astra,
    Gpt61Terra,
    ChatLatest,
    ClaudeFable51,
    ClaudeOpus55,
    ClaudeSonnet55,
    ClaudeHaiku55,
    Gemini4Argon,
    Gemini38Flash,
    Gemini35FlashLite,
    Gemini31FlashLite,
    Grok47,
    Grok46,
    Grok43
}

public static class AiModelExtensions
{
    public static string ToApiName(this AiModel model) => AiModelCatalog.Get(model).ApiName;

    public static string ToDisplayName(this AiModel model) => AiModelCatalog.Get(model).DisplayName;

    public static ProviderType GetProvider(this AiModel model) => AiModelCatalog.Get(model).Provider;

    public static bool SupportsThinking(this AiModel model) => AiModelCatalog.Get(model).Capabilities.SupportsThinking;

    public static int GetContextWindow(this AiModel model) => AiModelCatalog.Get(model).ContextWindow;
}
