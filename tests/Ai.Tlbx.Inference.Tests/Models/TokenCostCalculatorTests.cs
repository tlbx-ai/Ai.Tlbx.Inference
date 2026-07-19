namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class TokenCostCalculatorTests
{
    [Fact]
    public void Estimate_CalculatesProviderAndCustomerCost()
    {
        var usage = new TokenUsage
        {
            InputTokens = 1_000,
            CacheReadTokens = 100,
            CacheWriteTokens = 50,
            OutputTokens = 200,
            ThinkingTokens = 25
        };
        var rates = new TokenCostRates
        {
            ProviderId = "test",
            ModelApiName = "test-model",
            InputPerMillion = 10m,
            CachedInputPerMillion = 1m,
            CacheWriteInputPerMillion = 2m,
            OutputPerMillion = 20m
        };

        var estimate = TokenCostCalculator.Estimate(usage, rates, markupPercent: 100m);

        Assert.Equal("test", estimate.ProviderId);
        Assert.Equal("test-model", estimate.ModelApiName);
        Assert.Equal("USD", estimate.Currency);
        Assert.Equal(1_000, estimate.InputTokens);
        Assert.Equal(850, estimate.UncachedInputTokens);
        Assert.Equal(100, estimate.CacheReadTokens);
        Assert.Equal(50, estimate.CacheWriteTokens);
        Assert.Equal(225, estimate.OutputTokens);
        Assert.Equal(0.0132m, estimate.ProviderCost);
        Assert.Equal(0.0264m, estimate.CustomerCost);
        Assert.Equal(100m, estimate.MarkupPercent);
    }

    [Fact]
    public void EstimateCost_UsesModelCatalogRates()
    {
        var usage = new TokenUsage
        {
            InputTokens = 1_000,
            OutputTokens = 500
        };

        var estimate = usage.EstimateCost(AiModel.Gpt54, markupPercent: 100m);

        Assert.Equal("openai", estimate.ProviderId);
        Assert.Equal("gpt-5.4", estimate.ModelApiName);
        Assert.True(estimate.ProviderCost > 0m);
        Assert.Equal(estimate.ProviderCost * 2m, estimate.CustomerCost);
    }

    [Fact]
    public void AllAiModels_HaveCostRates()
    {
        foreach (var model in Enum.GetValues<AiModel>())
        {
            var rates = AiModelCostCatalog.GetRates(model);

            Assert.Equal(model.ToApiName(), rates.ModelApiName);
            Assert.False(string.IsNullOrWhiteSpace(rates.ProviderId));
        }
    }

    [Fact]
    public void OpenAiStandardRates_DoNotUseBatchDiscounts()
    {
        var gpt54 = AiModelCostCatalog.GetRates(AiModel.Gpt54);
        var gpt52Pro = AiModelCostCatalog.GetRates(AiModel.Gpt52Pro);

        Assert.Equal(2.5m, gpt54.InputPerMillion);
        Assert.Equal(0.25m, gpt54.CachedInputPerMillion);
        Assert.Equal(15m, gpt54.OutputPerMillion);
        Assert.Equal(21m, gpt52Pro.InputPerMillion);
        Assert.Null(gpt52Pro.CachedInputPerMillion);
        Assert.Equal(168m, gpt52Pro.OutputPerMillion);
    }

    [Fact]
    public void Estimate_RejectsNegativeMarkup()
    {
        var rates = new TokenCostRates
        {
            ProviderId = "test",
            ModelApiName = "test-model",
            InputPerMillion = 1m,
            OutputPerMillion = 1m
        };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TokenCostCalculator.Estimate(new TokenUsage(), rates, markupPercent: -1m));
    }
}
