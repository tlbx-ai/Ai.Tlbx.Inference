namespace Ai.Tlbx.Inference.IntegrationTests.Providers;

[Trait("Category", "Integration")]
public sealed class ProviderAttachmentSmokeTests
{
    public static IEnumerable<object[]> OpenAiModels()
        => IntegrationScenarioHelper.GetAttachmentModelsForProvider(ProviderType.OpenAi);

    public static IEnumerable<object[]> AnthropicModels()
        => IntegrationScenarioHelper.GetAttachmentModelsForProvider(ProviderType.Anthropic);

    public static IEnumerable<object[]> GoogleModels()
        => IntegrationScenarioHelper.GetAttachmentModelsForProvider(ProviderType.Google);

    public static IEnumerable<object[]> XaiModels()
        => IntegrationScenarioHelper.GetAttachmentModelsForProvider(ProviderType.Xai);

    [RequiresEnvironmentTheory("OPENAI_API_KEY")]
    [MemberData(nameof(OpenAiModels))]
    public async Task OpenAi_Attachment_ReturnsExpectedAnswer(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.OpenAi);
            await AssertAttachmentAsync(testClient.Client, model, ct);
        });
    }

    [RequiresEnvironmentTheory("ANTHROPIC_API_KEY")]
    [MemberData(nameof(AnthropicModels))]
    public async Task Anthropic_Attachment_ReturnsExpectedAnswer(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Anthropic);
            await AssertAttachmentAsync(testClient.Client, model, ct);
        });
    }

    [RequiresEnvironmentTheory("GOOGLE_API_KEY")]
    [MemberData(nameof(GoogleModels))]
    public async Task Google_Attachment_ReturnsExpectedAnswer(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
            await AssertAttachmentAsync(testClient.Client, model, ct);
        });
    }

    [RequiresEnvironmentFact("GOOGLE_API_KEY")]
    public async Task Google_Gemini35Flash_Attachment_ReturnsExpectedAnswer()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
            await AssertAttachmentAsync(testClient.Client, AiModel.Gemini35Flash, ct);
        });
    }

    [RequiresEnvironmentTheory("XAI_API_KEY")]
    [MemberData(nameof(XaiModels))]
    public async Task Xai_Attachment_ReturnsExpectedAnswer(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Xai);
            await AssertAttachmentAsync(testClient.Client, model, ct);
        });
    }

    private static async Task AssertAttachmentAsync(AiInferenceClient client, AiModel model, CancellationToken ct)
    {
        var response = await IntegrationScenarioHelper.ExecuteAttachmentSmokeAsync(client, model, ct);

        Assert.Contains("GREEN", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("42", response.Content, StringComparison.OrdinalIgnoreCase);
    }
}
