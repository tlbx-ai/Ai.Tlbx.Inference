namespace Ai.Tlbx.Inference.IntegrationTests.Providers;

[Trait("Category", "Integration")]
public sealed class ProviderSmokeTests
{
    public static IEnumerable<object[]> OpenAiModels()
        => IntegrationScenarioHelper.GetModelsForProvider(ProviderType.OpenAi);

    public static IEnumerable<object[]> AnthropicModels()
        => IntegrationScenarioHelper.GetModelsForProvider(ProviderType.Anthropic);

    public static IEnumerable<object[]> GoogleModels()
        => IntegrationScenarioHelper.GetModelsForProvider(ProviderType.Google);

    public static IEnumerable<object[]> XaiModels()
        => IntegrationScenarioHelper.GetModelsForProvider(ProviderType.Xai);

    [RequiresEnvironmentTheory("OPENAI_API_KEY")]
    [MemberData(nameof(OpenAiModels))]
    public async Task OpenAi_SimplePrompt_ReturnsContent(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.OpenAi);
            var response = await IntegrationScenarioHelper.ExecuteSmokeRequestAsync(testClient.Client, model, ct);
            Assert.False(string.IsNullOrWhiteSpace(response.Content));
        });
    }

    [RequiresEnvironmentTheory("ANTHROPIC_API_KEY")]
    [MemberData(nameof(AnthropicModels))]
    public async Task Anthropic_SimplePrompt_ReturnsContent(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Anthropic);
            var response = await IntegrationScenarioHelper.ExecuteSmokeRequestAsync(testClient.Client, model, ct);
            Assert.False(string.IsNullOrWhiteSpace(response.Content));
        });
    }

    [RequiresEnvironmentTheory("GOOGLE_API_KEY")]
    [MemberData(nameof(GoogleModels))]
    public async Task Google_SimplePrompt_ReturnsContent(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
            var response = await IntegrationScenarioHelper.ExecuteSmokeRequestAsync(testClient.Client, model, ct);
            Assert.False(string.IsNullOrWhiteSpace(response.Content));
        });
    }

    [RequiresEnvironmentFact("GOOGLE_API_KEY")]
    public async Task Google_Gemini35Flash_SimplePrompt_ReturnsContent()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
            var response = await IntegrationScenarioHelper.ExecuteSmokeRequestAsync(testClient.Client, AiModel.Gemini35Flash, ct);
            Assert.False(string.IsNullOrWhiteSpace(response.Content));
        });
    }

    [RequiresEnvironmentTheory("XAI_API_KEY")]
    [MemberData(nameof(XaiModels))]
    public async Task Xai_SimplePrompt_ReturnsContent(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Xai);
            var response = await IntegrationScenarioHelper.ExecuteSmokeRequestAsync(testClient.Client, model, ct);
            Assert.False(string.IsNullOrWhiteSpace(response.Content));
        });
    }
}
