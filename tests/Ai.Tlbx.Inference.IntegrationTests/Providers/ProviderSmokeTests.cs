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

    [RequiresEnvironmentFact("OPENAI_API_KEY")]
    public async Task OpenAi_Gpt56Luna_HighReasoningAnd64KOutput_ReturnsContent()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.OpenAi);
            var response = await testClient.Client.CompleteAsync(new CompletionRequest
            {
                Model = AiModel.Gpt56Luna,
                Messages = [new ChatMessage { Role = ChatRole.User, Content = "Antworte exakt mit: LUNA-OK" }],
                ThinkingBudget = 32_000,
                MaxTokens = 65_536,
            }, ct);

            Assert.Contains("LUNA-OK", response.Content, StringComparison.Ordinal);
            Assert.True(response.Usage.OutputTokens >= response.Usage.ThinkingTokens);
            Assert.Equal(response.Usage.InputTokens + response.Usage.OutputTokens, response.Usage.TotalTokens);
        });
    }

    [RequiresEnvironmentFact("OPENAI_API_KEY")]
    public async Task OpenAi_Gpt56Luna_Grounding_ReturnsCitedWebSourcesAndUsage()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.OpenAi);
            var response = await testClient.Client.CompleteAsync(new CompletionRequest
            {
                Model = AiModel.Gpt56Luna,
                Messages =
                [
                    new ChatMessage
                    {
                        Role = ChatRole.User,
                        Content = "Search the web for the German Wikipedia article about Bauhaus. Answer in one sentence and cite the source."
                    }
                ],
                ThinkingBudget = 8_000,
                MaxTokens = 16_384,
                Grounding = new GroundingOptions
                {
                    MaxSearches = 3,
                    AllowedDomains = ["wikipedia.org"],
                },
            }, ct);

            Assert.False(string.IsNullOrWhiteSpace(response.Content));
            Assert.NotNull(response.Grounding);
            Assert.True(response.Grounding!.Usage.WebSearchCalls > 0);
            Assert.Contains(response.Grounding.Sources, source =>
                source.Url.Contains("wikipedia.org", StringComparison.OrdinalIgnoreCase));
        });
    }

    [RequiresEnvironmentFact("OPENAI_API_KEY")]
    public async Task OpenAi_Gpt56Luna_ImageGrounding_ReturnsAttributedImage()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.OpenAi);
            var response = await testClient.Client.CompleteAsync(new CompletionRequest
            {
                Model = AiModel.Gpt56Luna,
                Messages =
                [
                    new ChatMessage
                    {
                        Role = ChatRole.User,
                        Content = "Find one relevant image of the Bauhaus building in Dessau for a school lesson. Briefly describe it and name its source."
                    }
                ],
                ThinkingBudget = 8_000,
                MaxTokens = 16_384,
                Grounding = new GroundingOptions
                {
                    EnableImageSearch = true,
                    MaxSearches = 3,
                },
            }, ct);

            Assert.False(string.IsNullOrWhiteSpace(response.Content));
            Assert.NotNull(response.Grounding);
            Assert.True(response.Grounding!.Usage.ImageSearchCalls > 0);
            Assert.NotEmpty(response.Grounding.Images);
            Assert.All(response.Grounding.Images, image => Assert.False(string.IsNullOrWhiteSpace(image.Url)));
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

    [RequiresEnvironmentFact("ANTHROPIC_API_KEY")]
    public async Task Anthropic_Haiku45_Grounding_ReturnsCitedSourceAndUsage()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Anthropic);
            var response = await testClient.Client.CompleteAsync(new CompletionRequest
            {
                Model = AiModel.ClaudeHaiku45,
                Messages = [new ChatMessage { Role = ChatRole.User, Content = "Search Wikipedia for Claude Shannon's birth year. Reply in one sentence with a citation." }],
                MaxTokens = 1_024,
                Grounding = new GroundingOptions { MaxSearches = 2, AllowedDomains = ["wikipedia.org"] },
            }, ct);

            Assert.False(string.IsNullOrWhiteSpace(response.Content));
            Assert.NotNull(response.Grounding);
            Assert.True(response.Grounding!.Usage.WebSearchCalls > 0);
            Assert.NotEmpty(response.Grounding.Sources);
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

    [RequiresEnvironmentFact("GOOGLE_API_KEY")]
    public async Task Google_Gemini35Flash_Grounding_ReturnsSourceAndBilledQueries()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Google);
            var response = await testClient.Client.CompleteAsync(new CompletionRequest
            {
                Model = AiModel.Gemini35Flash,
                Messages = [new ChatMessage { Role = ChatRole.User, Content = "Search the web for the German Wikipedia article about Bauhaus. Reply in one sentence with a source." }],
                MaxTokens = 2_048,
                Grounding = new GroundingOptions { MaxSearches = 2 },
            }, ct);

            Assert.False(string.IsNullOrWhiteSpace(response.Content));
            Assert.NotNull(response.Grounding);
            Assert.True(response.Grounding!.Usage.WebSearchCalls > 0);
            Assert.NotEmpty(response.Grounding.Sources);
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

    [RequiresEnvironmentFact("XAI_API_KEY")]
    public async Task Xai_Grok41Fast_Grounding_ReturnsCitedSourceAndUsage()
    {
        await IntegrationTestTimeout.ExecuteAsync(async ct =>
        {
            using var testClient = IntegrationScenarioHelper.CreateClient(ProviderType.Xai);
            var response = await testClient.Client.CompleteAsync(new CompletionRequest
            {
                Model = AiModel.Grok41FastNonReasoning,
                Messages = [new ChatMessage { Role = ChatRole.User, Content = "Search Wikipedia for the Bauhaus school. Reply in one sentence with a citation." }],
                MaxTokens = 1_024,
                Grounding = new GroundingOptions { MaxSearches = 2, AllowedDomains = ["wikipedia.org"] },
            }, ct);

            Assert.False(string.IsNullOrWhiteSpace(response.Content));
            Assert.NotNull(response.Grounding);
            Assert.True(response.Grounding!.Usage.WebSearchCalls > 0);
            Assert.NotEmpty(response.Grounding.Sources);
        });
    }
}
