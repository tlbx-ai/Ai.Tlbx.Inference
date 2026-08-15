namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class AiModelExtensionsTests
{
    [Theory]
    [InlineData(AiModel.Gpt52, "gpt-5.2")]
    [InlineData(AiModel.Gpt52Pro, "gpt-5.2-pro")]
    [InlineData(AiModel.Gpt52Chat, "gpt-5.2-chat-latest")]
    [InlineData(AiModel.Gpt53Chat, "gpt-5.3-chat-latest")]
    [InlineData(AiModel.Gpt54, "gpt-5.4")]
    [InlineData(AiModel.Gpt56Luna, "gpt-5.6-luna")]
    [InlineData(AiModel.ClaudeOpus46, "claude-opus-4-6")]
    [InlineData(AiModel.ClaudeSonnet46, "claude-sonnet-4-6")]
    [InlineData(AiModel.ClaudeHaiku45, "claude-haiku-4-5-20251001")]
    [InlineData(AiModel.Gemini35Flash, "gemini-3.5-flash")]
    [InlineData(AiModel.Gemini3FlashPreview, "gemini-3-flash-preview")]
    [InlineData(AiModel.Gemini31ProPreview, "gemini-3.1-pro-preview")]
    [InlineData(AiModel.Gemini31FlashLitePreview, "gemini-3.1-flash-lite-preview")]
    [InlineData(AiModel.Grok41Fast, "grok-4-1-fast-reasoning")]
    [InlineData(AiModel.Grok4, "grok-4")]
    public void ToApiName_ReturnsExpectedValue(AiModel model, string expected)
    {
        Assert.Equal(expected, model.ToApiName());
    }

    [Theory]
    [InlineData(AiModel.Gpt52, "GPT-5.2")]
    [InlineData(AiModel.Gpt54, "GPT-5.4")]
    [InlineData(AiModel.Gpt56Luna, "GPT-5.6 Luna")]
    [InlineData(AiModel.ClaudeOpus46, "Claude Opus 4.6")]
    [InlineData(AiModel.Gpt53Chat, "GPT-5.3 Chat")]
    [InlineData(AiModel.Gemini35Flash, "Gemini 3.5 Flash")]
    [InlineData(AiModel.Gemini31FlashLitePreview, "Gemini 3.1 Flash-Lite Preview")]
    [InlineData(AiModel.ClaudeSonnet46, "Claude Sonnet 4.6")]
    [InlineData(AiModel.Grok4, "Grok 4")]
    public void ToDisplayName_ReturnsExpectedValue(AiModel model, string expected)
    {
        Assert.Equal(expected, model.ToDisplayName());
    }

    [Theory]
    [InlineData(AiModel.Gpt52, ProviderType.OpenAi)]
    [InlineData(AiModel.Gpt53Chat, ProviderType.OpenAi)]
    [InlineData(AiModel.Gpt56Luna, ProviderType.OpenAi)]
    [InlineData(AiModel.ClaudeOpus46, ProviderType.Anthropic)]
    [InlineData(AiModel.ClaudeSonnet46, ProviderType.Anthropic)]
    [InlineData(AiModel.Gemini35Flash, ProviderType.Google)]
    [InlineData(AiModel.Gemini31ProPreview, ProviderType.Google)]
    [InlineData(AiModel.Gemini3FlashPreview, ProviderType.Google)]
    [InlineData(AiModel.Grok4, ProviderType.Xai)]
    public void GetProvider_ReturnsCorrectProviderType(AiModel model, ProviderType expected)
    {
        Assert.Equal(expected, model.GetProvider());
    }

    [Theory]
    [InlineData(AiModel.ClaudeOpus46, true)]
    [InlineData(AiModel.ClaudeSonnet46, true)]
    [InlineData(AiModel.Gemini35Flash, true)]
    [InlineData(AiModel.Gemini31ProPreview, true)]
    [InlineData(AiModel.Grok41Fast, true)]
    [InlineData(AiModel.Gpt56Luna, true)]
    [InlineData(AiModel.Gpt52, false)]
    [InlineData(AiModel.Gpt54, false)]
    [InlineData(AiModel.ClaudeHaiku45, false)]
    [InlineData(AiModel.Gemini31FlashLitePreview, false)]
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
