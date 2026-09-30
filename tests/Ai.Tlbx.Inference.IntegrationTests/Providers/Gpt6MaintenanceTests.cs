using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;

namespace Ai.Tlbx.Inference.IntegrationTests.Providers;

public sealed class Gpt6MaintenanceTests
{
    [RequiresEnvironmentTheory("OPENAI_API_KEY")]
    [InlineData(AiModel.Gpt6Luna)]
    [InlineData(AiModel.Gpt6Sol)]
    [InlineData(AiModel.Gpt61Sol)]
    public async Task HighFast_ExecutesToolAndReturnsVerifiedAnswer(AiModel model)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(3) };
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenAi(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!));
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{},"required":[],"additionalProperties":false}""");
        var executions = 0;
        var result = await client.CompleteWithToolsAsync(new CompletionRequest
        {
            Model = model, ReasoningEffort = "high", ServiceTier = "fast", MaxTokens = 4096,
            SystemMessage = "Call calculate exactly once to compute 17 * 23. Then reply only with the number returned by the tool.",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "What is 17 * 23?" }]
        }, [new ToolDefinition { Name = "calculate", Description = "Returns the exact product of 17 and 23.", ParametersSchema = schema.RootElement }],
            call =>
            {
                executions++;
                return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "391" });
            }, maxIterations: 3);
        Assert.Equal(1, executions);
        Assert.Contains("391", result.Content);
        var rates = AiModelCostCatalog.GetRates(model, "fast");
        var cost = TokenCostCalculator.Estimate(result.Usage, rates).ProviderCost;
        Console.WriteLine($"{model}: input={result.Usage.InputTokens}; output={result.Usage.OutputTokens}; tier={result.Usage.ServiceTier}; costUSD={cost}");
    }
}
