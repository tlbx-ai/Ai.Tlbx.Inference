namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class AiModelCatalogTests
{
    [Fact]
    public void AllModels_ArePresentInCatalog()
    {
        foreach (var model in Enum.GetValues<AiModel>())
        {
            var descriptor = AiModelCatalog.Get(model);
            Assert.Equal(model, descriptor.Model);
        }
    }

    [Fact]
    public void Gpt52Pro_PrefersResponsesApi()
    {
        var descriptor = AiModelCatalog.Get(AiModel.Gpt52Pro);

        Assert.Equal(ModelEndpointFamily.Responses, descriptor.PreferredEndpoint);
        Assert.False(descriptor.Capabilities.SupportsChatCompletionsApi);
        Assert.True(descriptor.Capabilities.SupportsResponsesApi);
    }

    [Fact]
    public void Gpt56Luna_PrefersResponsesApiForReasoningToolCalls()
    {
        var descriptor = AiModelCatalog.Get(AiModel.Gpt56Luna);

        Assert.Equal(ModelEndpointFamily.Responses, descriptor.PreferredEndpoint);
        Assert.True(descriptor.Capabilities.SupportsChatCompletionsApi);
        Assert.True(descriptor.Capabilities.SupportsResponsesApi);
    }

    [Fact]
    public void GoogleCatalogEntries_AreGroupedByProvider()
    {
        var models = AiModelCatalog.GetByProvider(ProviderType.Google);

        Assert.NotEmpty(models);
        Assert.All(models, model => Assert.Equal(ProviderType.Google, model.Provider));
    }

}
