using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class MaintenanceCatalogTests
{
    [Fact]
    public void ExistingEnumNumbersRemainStable()
    {
        Assert.Equal(0, (int)AiModel.Gpt52);
        Assert.Equal(6, (int)AiModel.ClaudeOpus46);
        Assert.Equal(16, (int)AiModel.Gpt6Luna);
        Assert.Equal(18, (int)AiModel.Gpt61Sol);
        Assert.Equal(2, (int)EmbeddingModel.GeminiEmbedding001);
        Assert.Equal(4, (int)ImageGenerationModel.GptImage1Mini);
    }

    [Theory]
    [InlineData(AiModel.Gpt6Astra, "none", null)]
    [InlineData(AiModel.ClaudeOpus55, "none", null)]
    [InlineData(AiModel.ClaudeHaiku55, "max", null)]
    [InlineData(AiModel.Gemini38Flash, "max", null)]
    [InlineData(AiModel.Grok47, "none", null)]
    [InlineData(AiModel.Grok47, "max", null)]
    [InlineData(AiModel.Gemini38Flash, "low", "fast")]
    [InlineData(AiModel.Gpt61Terra, "low", "ultrafast")]
    public async Task UnsupportedSettingsFailBeforeHttp(AiModel model, string effort, string? tier)
    {
        using var http = new HttpClient(new MockHttpHandler(_ => throw new InvalidOperationException("No network expected")));
        var options = new AiInferenceOptions().AddOpenAi("test").AddAnthropic("test").AddGoogle("test").AddXai("test");
        var client = new AiInferenceClient(http, options);
        await Assert.ThrowsAsync<ArgumentException>(() => client.CompleteAsync(new CompletionRequest
        {
            Model = model, Messages = [], ReasoningEffort = effort, ServiceTier = tier
        }));
    }

    [Theory]
    [InlineData(AiModel.ClaudeHaiku55, 100000, 0.0105)]
    [InlineData(AiModel.ClaudeHaiku55, 100001, 0.0525005)]
    [InlineData(AiModel.Grok47, 199999, 0.405998)]
    [InlineData(AiModel.Grok47, 200000, 0.812)]
    public void ContextPriceBoundaryAppliesToEntireRequest(AiModel model, int input, double expected)
    {
        var estimate = new TokenUsage { InputTokens = input, OutputTokens = 1000 }.EstimateCost(model);
        Assert.Equal((decimal)expected, estimate.ProviderCost);
    }

    [Fact]
    public void PromotionalPricesExpireOnProviderDates()
    {
        Assert.Equal(2m, AiModelCostCatalog.GetRates(AiModel.Gemini4Argon, new DateOnly(2026, 11, 30)).InputPerMillion);
        Assert.Equal(4m, AiModelCostCatalog.GetRates(AiModel.Gemini4Argon, new DateOnly(2026, 12, 1)).InputPerMillion);
        Assert.Equal(0.75m, AiModelCostCatalog.GetRates(AiModel.Gemini38Flash, new DateOnly(2026, 12, 31)).InputPerMillion);
        Assert.Equal(1.5m, AiModelCostCatalog.GetRates(AiModel.Gemini38Flash, new DateOnly(2027, 1, 1)).InputPerMillion);
    }

    [Fact]
    public void UltrafastPricingUsesSixTimesStandardRates()
    {
        Assert.Equal(60m, AiModelCostCatalog.GetRates(AiModel.Gpt6Astra, "ultrafast").InputPerMillion);
        Assert.Equal(60m, AiModelCostCatalog.GetRates(AiModel.Gpt61Sol, "ultrafast").OutputPerMillion);
    }
}
