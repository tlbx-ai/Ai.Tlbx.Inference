namespace Ai.Tlbx.Inference.IntegrationTests.Providers;

[Trait("Category", "Integration")]
public sealed class ProviderStreamingSmokeTests
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
    public async Task OpenAi_Stream_ReturnsMoreThanOneChunk(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.OpenAi);
            await AssertStreamingAsync(testClient.Client, model, ct);
        });
    }

    [RequiresEnvironmentTheory("ANTHROPIC_API_KEY")]
    [MemberData(nameof(AnthropicModels))]
    public async Task Anthropic_Stream_ReturnsMoreThanOneChunk(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Anthropic);
            await AssertStreamingAsync(testClient.Client, model, ct);
        });
    }

    [RequiresEnvironmentTheory("GOOGLE_API_KEY")]
    [MemberData(nameof(GoogleModels))]
    public async Task Google_Stream_ReturnsMoreThanOneChunk(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
            await AssertStreamingAsync(testClient.Client, model, ct);
        });
    }

    [RequiresEnvironmentTheory("XAI_API_KEY")]
    [MemberData(nameof(XaiModels))]
    public async Task Xai_Stream_ReturnsMoreThanOneChunk(AiModel model)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Xai);
            await AssertStreamingAsync(testClient.Client, model, ct);
        });
    }

    private static async Task AssertStreamingAsync(AiInferenceClient client, AiModel model, CancellationToken ct)
    {
        var chunks = await IntegrationScenarioHelper.ExecuteStreamingSmokeAsync(client, model, ct);
        var combined = string.Concat(chunks);

        Assert.True(chunks.Count > 1, $"Expected multiple stream chunks for {model}, but got {chunks.Count}.");
        Assert.True(combined.Length >= 80, $"Expected a substantial streamed response for {model}, but only received {combined.Length} characters.");
        Assert.Contains('\n', combined);
    }
}
