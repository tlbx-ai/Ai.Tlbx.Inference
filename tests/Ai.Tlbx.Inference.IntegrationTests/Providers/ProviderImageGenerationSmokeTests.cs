namespace Ai.Tlbx.Inference.IntegrationTests.Providers;

[Trait("Category", "Integration")]
public sealed class ProviderImageGenerationSmokeTests
{
    [RequiresEnvironmentTheory("GOOGLE_API_KEY")]
    [InlineData(ProviderType.Google)]
    public async Task Google_ImageGeneration_ReturnsValidPng(ProviderType providerType)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(providerType);
            var result = await IntegrationScenarioHelper.ExecuteGoogleImageGenerationSmokeAsync(testClient.Client, ct);

            Assert.True(result.ImageBytes.Length > 8, "Google image generation returned too few bytes.");
            Assert.True(IntegrationScenarioHelper.IsPng(result.ImageBytes), $"Expected a PNG signature. Artifact: {result.ArtifactPath}");
        });
    }

    [RequiresEnvironmentTheory("OPENAI_API_KEY")]
    [InlineData(ProviderType.OpenAi)]
    public async Task OpenAi_ImageGeneration_ReturnsValidPng(ProviderType providerType)
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(providerType);
            var result = await IntegrationScenarioHelper.ExecuteOpenAiImageGenerationSmokeAsync(testClient.Client, ct);

            Assert.True(result.ImageBytes.Length > 8, "OpenAI image generation returned too few bytes.");
            Assert.True(IntegrationScenarioHelper.IsPng(result.ImageBytes), $"Expected a PNG signature. Artifact: {result.ArtifactPath}");
        });
    }
}
