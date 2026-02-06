namespace Ai.Tlbx.Inference;

public enum AiModel
{
    // OpenAI
    Gpt52,
    Gpt52Pro,
    Gpt52Chat,
    Gpt53Codex,
    Gpt5,
    O3,
    O3Pro,
    O4Mini,

    // Anthropic
    ClaudeOpus46,
    ClaudeOpus45,
    ClaudeSonnet45,
    ClaudeHaiku45,

    // Google
    Gemini3ProPreview,
    Gemini3FlashPreview,
    Gemini25Pro,
    Gemini25Flash,

    // xAI
    Grok41Fast,
    Grok4,
    Grok3
}

public static class AiModelExtensions
{
    public static string ToApiName(this AiModel model) => model switch
    {
        AiModel.Gpt52 => "gpt-5.2",
        AiModel.Gpt52Pro => "gpt-5.2-pro",
        AiModel.Gpt52Chat => "gpt-5.2-chat-latest",
        AiModel.Gpt53Codex => "gpt-5.3-codex",
        AiModel.Gpt5 => "gpt-5",
        AiModel.O3 => "o3",
        AiModel.O3Pro => "o3-pro",
        AiModel.O4Mini => "o4-mini",
        AiModel.ClaudeOpus46 => "claude-opus-4-6",
        AiModel.ClaudeOpus45 => "claude-opus-4-5-20250220",
        AiModel.ClaudeSonnet45 => "claude-sonnet-4-5-20250929",
        AiModel.ClaudeHaiku45 => "claude-haiku-4-5-20251001",
        AiModel.Gemini3ProPreview => "gemini-3-pro-preview",
        AiModel.Gemini3FlashPreview => "gemini-3-flash-preview",
        AiModel.Gemini25Pro => "gemini-2.5-pro",
        AiModel.Gemini25Flash => "gemini-2.5-flash",
        AiModel.Grok41Fast => "grok-4-1-fast-reasoning",
        AiModel.Grok4 => "grok-4",
        AiModel.Grok3 => "grok-3-beta",
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static string ToDisplayName(this AiModel model) => model switch
    {
        AiModel.Gpt52 => "GPT-5.2",
        AiModel.Gpt52Pro => "GPT-5.2 Pro",
        AiModel.Gpt52Chat => "GPT-5.2 Chat",
        AiModel.Gpt53Codex => "GPT-5.3 Codex",
        AiModel.Gpt5 => "GPT-5",
        AiModel.O3 => "o3",
        AiModel.O3Pro => "o3 Pro",
        AiModel.O4Mini => "o4 Mini",
        AiModel.ClaudeOpus46 => "Claude Opus 4.6",
        AiModel.ClaudeOpus45 => "Claude Opus 4.5",
        AiModel.ClaudeSonnet45 => "Claude Sonnet 4.5",
        AiModel.ClaudeHaiku45 => "Claude Haiku 4.5",
        AiModel.Gemini3ProPreview => "Gemini 3 Pro Preview",
        AiModel.Gemini3FlashPreview => "Gemini 3 Flash Preview",
        AiModel.Gemini25Pro => "Gemini 2.5 Pro",
        AiModel.Gemini25Flash => "Gemini 2.5 Flash",
        AiModel.Grok41Fast => "Grok 4.1 Fast",
        AiModel.Grok4 => "Grok 4",
        AiModel.Grok3 => "Grok 3",
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static ProviderType GetProvider(this AiModel model) => model switch
    {
        AiModel.Gpt52 or AiModel.Gpt52Pro or AiModel.Gpt52Chat or
        AiModel.Gpt53Codex or AiModel.Gpt5 or
        AiModel.O3 or AiModel.O3Pro or AiModel.O4Mini => ProviderType.OpenAi,

        AiModel.ClaudeOpus46 or AiModel.ClaudeOpus45 or
        AiModel.ClaudeSonnet45 or AiModel.ClaudeHaiku45 => ProviderType.Anthropic,

        AiModel.Gemini3ProPreview or AiModel.Gemini3FlashPreview or
        AiModel.Gemini25Pro or AiModel.Gemini25Flash => ProviderType.Google,

        AiModel.Grok41Fast or AiModel.Grok4 or AiModel.Grok3 => ProviderType.Xai,

        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static bool SupportsThinking(this AiModel model) => model switch
    {
        AiModel.O3 or AiModel.O3Pro or AiModel.O4Mini or
        AiModel.ClaudeOpus46 or AiModel.ClaudeOpus45 or AiModel.ClaudeSonnet45 or
        AiModel.Gemini25Pro or AiModel.Gemini25Flash or
        AiModel.Gemini3ProPreview or AiModel.Gemini3FlashPreview or
        AiModel.Grok41Fast or AiModel.Grok4 => true,
        _ => false
    };

    public static int GetContextWindow(this AiModel model) => model switch
    {
        AiModel.Gpt52 or AiModel.Gpt52Pro => 400000,
        AiModel.Gpt52Chat => 128000,
        AiModel.Gpt53Codex => 400000,
        AiModel.Gpt5 => 400000,
        AiModel.O3 or AiModel.O3Pro or AiModel.O4Mini => 200000,
        AiModel.ClaudeOpus46 or AiModel.ClaudeOpus45 or
        AiModel.ClaudeSonnet45 or AiModel.ClaudeHaiku45 => 200000,
        AiModel.Gemini3ProPreview or AiModel.Gemini3FlashPreview or
        AiModel.Gemini25Pro or AiModel.Gemini25Flash => 1000000,
        AiModel.Grok41Fast => 2000000,
        AiModel.Grok4 => 256000,
        AiModel.Grok3 => 131000,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };
}
