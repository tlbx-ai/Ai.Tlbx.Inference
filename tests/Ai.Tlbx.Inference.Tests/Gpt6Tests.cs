using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests;

public sealed class Gpt6Tests
{
    [Theory]
    [InlineData(AiModel.Gpt6Luna, "gpt-6-luna", 0.2, 0.02, 1)]
    [InlineData(AiModel.Gpt6Sol, "gpt-6-sol", 4, 0.4, 20)]
    [InlineData(AiModel.Gpt61Sol, "gpt-6.1-sol", 4, 0.2, 20)]
    public async Task Facade_UsesResponsesWithHighReasoningAndFastTier(
        AiModel model, string apiName, double inputRate, double cacheRate, double outputRate)
    {
        var calls = 0;
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            calls++;
            Assert.Equal("/v1/responses", request.RequestUri!.AbsolutePath);
            using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            Assert.Equal(apiName, body.RootElement.GetProperty("model").GetString());
            Assert.Equal("high", body.RootElement.GetProperty("reasoning").GetProperty("effort").GetString());
            Assert.Equal("fast", body.RootElement.GetProperty("service_tier").GetString());
            Assert.Equal("function", body.RootElement.GetProperty("tools")[0].GetProperty("type").GetString());
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""
                    {"status":"completed","service_tier":"default","output":[{"type":"message",
                    "content":[{"type":"output_text","text":"42"}]}],
                    "usage":{"input_tokens":100,"output_tokens":50,"output_tokens_details":{"reasoning_tokens":30}}}
                    """)
            };
        }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenAi("test"));
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{},"additionalProperties":false}""");
        var response = await client.CompleteWithToolsAsync(new CompletionRequest
        {
            Model = model, ReasoningEffort = "high", ServiceTier = "fast", ThinkingBudget = 16000,
            MaxTokens = 16384, Messages = [new ChatMessage { Role = ChatRole.User, Content = "Calculate." }]
        }, [new ToolDefinition { Name = "calculate", Description = "Calculate", ParametersSchema = schema.RootElement }],
            call => Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "42" }));
        Assert.Equal(1, calls);
        Assert.Equal("42", response.Content);
        Assert.Equal("default", response.Usage.ServiceTier);
        Assert.Equal(30, response.Usage.ThinkingTokens);
        var rates = AiModelCostCatalog.GetRates(model, "fast");
        Assert.Equal((decimal)inputRate, rates.InputPerMillion);
        Assert.Equal((decimal)cacheRate, rates.CachedInputPerMillion);
        Assert.Equal((decimal)outputRate, rates.OutputPerMillion);
        Assert.Equal(ModelEndpointFamily.Responses, AiModelCatalog.Get(model).PreferredEndpoint);
    }

    [Fact]
    public async Task Sol61_RejectsNoReasoningBeforeCallingProvider()
    {
        using var http = new HttpClient(new MockHttpHandler(_ => throw new InvalidOperationException("Must not call provider")));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenAi("test"));
        await Assert.ThrowsAsync<ArgumentException>(() => client.CompleteAsync(new CompletionRequest
        {
            Model = AiModel.Gpt61Sol, ReasoningEffort = "none", Messages = []
        }));
    }
}
