namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class AiModelExtensionsTests
{
    [Theory]
    [InlineData(AiModel.Gpt52, "gpt-5.2")]
    [InlineData(AiModel.Gpt52Pro, "gpt-5.2-pro")]
    [InlineData(AiModel.Gpt52Chat, "gpt-5.2-chat-latest")]
    [InlineData(AiModel.Gpt53Codex, "gpt-5.3-codex")]
    [InlineData(AiModel.Gpt5, "gpt-5")]
    [InlineData(AiModel.O3, "o3")]
    [InlineData(AiModel.O3Pro, "o3-pro")]
    [InlineData(AiModel.O4Mini, "o4-mini")]
    [InlineData(AiModel.ClaudeOpus46, "claude-opus-4-6")]
    [InlineData(AiModel.ClaudeOpus45, "claude-opus-4-5-20250220")]
    [InlineData(AiModel.ClaudeSonnet45, "claude-sonnet-4-5-20250929")]
    [InlineData(AiModel.ClaudeHaiku45, "claude-haiku-4-5-20251001")]
    [InlineData(AiModel.Gemini3ProPreview, "gemini-3-pro-preview")]
    [InlineData(AiModel.Gemini3FlashPreview, "gemini-3-flash-preview")]
    [InlineData(AiModel.Gemini25Pro, "gemini-2.5-pro")]
    [InlineData(AiModel.Gemini25Flash, "gemini-2.5-flash")]
    [InlineData(AiModel.Grok41Fast, "grok-4-1-fast-reasoning")]
    [InlineData(AiModel.Grok4, "grok-4")]
    [InlineData(AiModel.Grok3, "grok-3-beta")]
    public void ToApiName_ReturnsExpectedValue(AiModel model, string expected)
    {
        Assert.Equal(expected, model.ToApiName());
    }

    [Theory]
    [InlineData(AiModel.Gpt52, "GPT-5.2")]
    [InlineData(AiModel.ClaudeOpus46, "Claude Opus 4.6")]
    [InlineData(AiModel.Gemini25Pro, "Gemini 2.5 Pro")]
    [InlineData(AiModel.Grok4, "Grok 4")]
    public void ToDisplayName_ReturnsExpectedValue(AiModel model, string expected)
    {
        Assert.Equal(expected, model.ToDisplayName());
    }

    [Theory]
    [InlineData(AiModel.Gpt52, ProviderType.OpenAi)]
    [InlineData(AiModel.O3, ProviderType.OpenAi)]
    [InlineData(AiModel.ClaudeOpus46, ProviderType.Anthropic)]
    [InlineData(AiModel.ClaudeSonnet45, ProviderType.Anthropic)]
    [InlineData(AiModel.Gemini25Pro, ProviderType.Google)]
    [InlineData(AiModel.Gemini3FlashPreview, ProviderType.Google)]
    [InlineData(AiModel.Grok4, ProviderType.Xai)]
    [InlineData(AiModel.Grok3, ProviderType.Xai)]
    public void GetProvider_ReturnsCorrectProviderType(AiModel model, ProviderType expected)
    {
        Assert.Equal(expected, model.GetProvider());
    }

    [Theory]
    [InlineData(AiModel.O3, true)]
    [InlineData(AiModel.ClaudeOpus46, true)]
    [InlineData(AiModel.ClaudeSonnet45, true)]
    [InlineData(AiModel.Gemini25Pro, true)]
    [InlineData(AiModel.Grok41Fast, true)]
    [InlineData(AiModel.Gpt52, false)]
    [InlineData(AiModel.ClaudeHaiku45, false)]
    [InlineData(AiModel.Grok3, false)]
    public void SupportsThinking_ReturnsExpectedValue(AiModel model, bool expected)
    {
        Assert.Equal(expected, model.SupportsThinking());
    }

    [Fact]
    public void AllModels_HaveApiName()
    {
        foreach (var model in Enum.GetValues<AiModel>())
        {
            var name = model.ToApiName();
            Assert.False(string.IsNullOrEmpty(name));
        }
    }

    [Fact]
    public void AllModels_HaveDisplayName()
    {
        foreach (var model in Enum.GetValues<AiModel>())
        {
            var name = model.ToDisplayName();
            Assert.False(string.IsNullOrEmpty(name));
        }
    }

    [Fact]
    public void AllModels_HaveProvider()
    {
        foreach (var model in Enum.GetValues<AiModel>())
        {
            var provider = model.GetProvider();
            Assert.True(Enum.IsDefined(provider));
        }
    }

    [Fact]
    public void AllModels_HaveContextWindow()
    {
        foreach (var model in Enum.GetValues<AiModel>())
        {
            var contextWindow = model.GetContextWindow();
            Assert.True(contextWindow > 0);
        }
    }
}
