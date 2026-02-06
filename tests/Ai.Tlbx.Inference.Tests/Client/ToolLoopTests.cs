using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Client;

public sealed class ToolLoopTests
{
    private static readonly IReadOnlyList<ToolDefinition> _testTools =
    [
        new ToolDefinition
        {
            Name = "get_weather",
            Description = "Get weather for a city",
            ParametersSchema = JsonDocument.Parse("""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""").RootElement,
        }
    ];

    [Fact]
    public async Task CompleteWithToolsAsync_SingleToolCall_ReturnsResult()
    {
        var callCount = 0;
        var handler = new MockHttpHandler(async _ =>
        {
            callCount++;
            await Task.CompletedTask;

            if (callCount == 1)
            {
                return BuildToolCallResponse("call_1", "get_weather", """{"city":"London"}""");
            }

            return BuildFinalResponse("The weather in London is sunny.");
        });

        var client = CreateClient(handler);

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "What's the weather in London?" }],
        };

        var result = await client.CompleteWithToolsAsync<string>(
            request,
            _testTools,
            tc =>
            {
                Assert.Equal("get_weather", tc.Name);
                return Task.FromResult(new ToolCallResult
                {
                    ToolCallId = tc.Id,
                    Result = """{"temp":"22C","condition":"sunny"}""",
                });
            });

        Assert.Equal("The weather in London is sunny.", result.Content);
        Assert.Equal(2, result.Iterations);
    }

    [Fact]
    public async Task CompleteWithToolsAsync_MultiTurn_AccumulatesMessages()
    {
        var callCount = 0;
        var handler = new MockHttpHandler(async _ =>
        {
            callCount++;
            await Task.CompletedTask;

            return callCount switch
            {
                1 => BuildToolCallResponse("call_1", "get_weather", """{"city":"London"}"""),
                2 => BuildToolCallResponse("call_2", "get_weather", """{"city":"Paris"}"""),
                _ => BuildFinalResponse("London is sunny, Paris is rainy."),
            };
        });

        var client = CreateClient(handler);

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Compare weather" }],
        };

        var result = await client.CompleteWithToolsAsync<string>(
            request,
            _testTools,
            tc => Task.FromResult(new ToolCallResult
            {
                ToolCallId = tc.Id,
                Result = """{"temp":"20C"}""",
            }));

        Assert.Equal("London is sunny, Paris is rainy.", result.Content);
        Assert.Equal(3, result.Iterations);
        Assert.True(result.Messages.Count > 1);
    }

    [Fact]
    public async Task CompleteWithToolsAsync_MaxIterationsExceeded_Throws()
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return BuildToolCallResponse("call_x", "get_weather", """{"city":"X"}""");
        });

        var client = CreateClient(handler);

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Loop" }],
        };

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => client.CompleteWithToolsAsync<string>(
                request,
                _testTools,
                tc => Task.FromResult(new ToolCallResult { ToolCallId = tc.Id, Result = "ok" }),
                maxIterations: 2));
    }

    [Fact]
    public async Task CompleteWithToolsAsync_AccumulatesTokenUsage()
    {
        var callCount = 0;
        var handler = new MockHttpHandler(async _ =>
        {
            callCount++;
            await Task.CompletedTask;

            if (callCount == 1)
            {
                return BuildToolCallResponse("call_1", "get_weather", """{"city":"A"}""", promptTokens: 50, completionTokens: 20);
            }

            return BuildFinalResponse("Done", promptTokens: 80, completionTokens: 30);
        });

        var client = CreateClient(handler);

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Do it" }],
        };

        var result = await client.CompleteWithToolsAsync<string>(
            request,
            _testTools,
            tc => Task.FromResult(new ToolCallResult { ToolCallId = tc.Id, Result = "ok" }));

        Assert.Equal(130, result.Usage.InputTokens);
        Assert.Equal(50, result.Usage.OutputTokens);
    }

    [Fact]
    public async Task CompleteWithToolsAsync_NoToolCallsOnFirstResponse_ReturnsDirect()
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return BuildFinalResponse("No tools needed.");
        });

        var client = CreateClient(handler);

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
        };

        var result = await client.CompleteWithToolsAsync<string>(
            request,
            _testTools,
            _ => throw new InvalidOperationException("Should not be called"));

        Assert.Equal("No tools needed.", result.Content);
        Assert.Equal(1, result.Iterations);
    }

    [Fact]
    public async Task CompleteWithToolsAsync_DeserializesTypedResult()
    {
        var callCount = 0;
        var handler = new MockHttpHandler(async _ =>
        {
            callCount++;
            await Task.CompletedTask;

            if (callCount == 1)
            {
                return BuildToolCallResponse("call_1", "get_weather", """{"city":"London"}""");
            }

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""
                {
                    "choices": [{
                        "message": { "content": "{\"city\":\"London\",\"temperature\":22}" },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
                }
                """, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var client = CreateClient(handler);

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather?" }],
        };

        var result = await client.CompleteWithToolsAsync<WeatherResult>(
            request,
            _testTools,
            tc => Task.FromResult(new ToolCallResult { ToolCallId = tc.Id, Result = "ok" }));

        Assert.Equal("London", result.Content.City);
        Assert.Equal(22, result.Content.Temperature);
    }

    private static AiInferenceClient CreateClient(MockHttpHandler handler)
    {
        var httpClient = new HttpClient(handler);
        var options = new AiInferenceOptions();
        options.AddOpenAi("test-key");
        return new AiInferenceClient(httpClient, options);
    }

    private static HttpResponseMessage BuildToolCallResponse(
        string callId,
        string name,
        string arguments,
        int promptTokens = 10,
        int completionTokens = 5)
    {
        var json = $$"""
        {
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "{{callId}}",
                        "type": "function",
                        "function": { "name": "{{name}}", "arguments": "{{arguments.Replace("\"", "\\\"")}}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": {{promptTokens}}, "completion_tokens": {{completionTokens}} }
        }
        """;

        return new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
        };
    }

    private static HttpResponseMessage BuildFinalResponse(
        string content,
        int promptTokens = 10,
        int completionTokens = 5)
    {
        var json = $$"""
        {
            "choices": [{ "message": { "content": "{{content}}" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": {{promptTokens}}, "completion_tokens": {{completionTokens}} }
        }
        """;

        return new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
        };
    }

    public sealed class WeatherResult
    {
        public string City { get; set; } = "";
        public int Temperature { get; set; }
    }
}
