using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;
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

        var result = await client.CompleteWithToolsAsync(
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

        var result = await client.CompleteWithToolsAsync(
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
            () => client.CompleteWithToolsAsync(
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

        var result = await client.CompleteWithToolsAsync(
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

        var result = await client.CompleteWithToolsAsync(
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
            JsonSchema = """{"type":"object","properties":{"city":{"type":"string"},"temperature":{"type":"integer"}},"required":["city","temperature"]}""",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather?" }],
        };

        var result = await client.CompleteWithToolsAsync(
            request,
            _testTools,
            tc => Task.FromResult(new ToolCallResult { ToolCallId = tc.Id, Result = "ok" }),
            ToolLoopJsonContext.Default.WeatherResult);

        Assert.Equal("London", result.Content.City);
        Assert.Equal(22, result.Content.Temperature);
    }

    [Fact]
    public async Task CompleteWithToolsAsync_AccumulatesGroundingAcrossIterations()
    {
        var callCount = 0;
        var handler = new MockHttpHandler(async _ =>
        {
            callCount++;
            await Task.CompletedTask;
            var json = callCount == 1
                ? """
                  {"status":"completed","output":[
                    {"type":"web_search_call","action":{"type":"search","query":"London weather","sources":[{"url":"https://weather.example/london","title":"London"}]}},
                    {"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"London\"}"}
                  ],"usage":{"input_tokens":10,"output_tokens":5}}
                  """
                : """
                  {"status":"completed","output":[
                    {"type":"web_search_call","action":{"type":"search","query":"Paris weather","sources":[{"url":"https://weather.example/paris","title":"Paris"}]}},
                    {"type":"message","content":[{"type":"output_text","text":"Done"}]}
                  ],"usage":{"input_tokens":20,"output_tokens":10}}
                  """;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
            };
        });
        var client = CreateClient(handler);

        var result = await client.CompleteWithToolsAsync(
            new CompletionRequest
            {
                Model = AiModel.Gpt52,
                Messages = [new ChatMessage { Role = ChatRole.User, Content = "Compare weather" }],
                Grounding = new GroundingOptions(),
            },
            _testTools,
            call => Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "sunny" }));

        Assert.NotNull(result.Grounding);
        Assert.Equal(2, result.Grounding!.Sources.Count);
        Assert.Equal(2, result.Grounding.Usage.WebSearchCalls);
        Assert.Equal(2, result.Iterations);
    }

    [Fact]
    public async Task ObserverCanDenyNextCallAfterRecordingFirstUsageAndToolResult()
    {
        var calls = 0;
        var observer = new RecordingObserver { DenyIteration = 2 };
        var client = CreateClient(new MockHttpHandler(_ =>
        {
            calls++;
            return Task.FromResult(BuildToolCallResponse("call_1", "get_weather", "{}"));
        }));

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.CompleteWithToolsAsync(
            ObservedRequest(observer), _testTools, call =>
            {
                Assert.Single(observer.Finished);
                return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "sunny" });
            }));

        Assert.Equal(1, calls);
        Assert.Equal(2, observer.Started.Count);
        Assert.Single(observer.Finished);
        Assert.Equal(10, observer.Finished[0].Usage!.Value.InputTokens);
        Assert.True(observer.Finished[0].ResponseCompleted);
        Assert.Equal("sunny", observer.Started[1].Request.Messages[^1].Content);
        Assert.Equal(3, observer.Started[1].Request.Messages.Count);
        Assert.Single(observer.Started[1].Tools);
        Assert.Null(observer.Started[1].Request.ToolIterationObserver);
        Assert.Single(observer.Started[0].Request.Messages);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ObserverRetainsUsageWhenLaterProviderOrToolFails(bool failTool)
    {
        var calls = 0;
        var observer = new RecordingObserver();
        var client = CreateClient(new MockHttpHandler(_ => Task.FromResult(++calls == 1
            ? BuildToolCallResponse("call_1", "get_weather", "{}")
            : new HttpResponseMessage(HttpStatusCode.BadRequest) { Content = new StringContent("rejected") })));

        await Assert.ThrowsAnyAsync<Exception>(() => client.CompleteWithToolsAsync(
            ObservedRequest(observer), _testTools, call => failTool
                ? throw new InvalidOperationException("tool failed")
                : Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "sunny" })));

        Assert.Equal(failTool ? 1 : 2, calls);
        Assert.Equal(calls, observer.Finished.Count);
        Assert.True(observer.Finished[0].ResponseCompleted);
        Assert.Equal(15, observer.Finished[0].Usage!.Value.TotalTokens);
        if (!failTool)
        {
            Assert.False(observer.Finished[1].ResponseCompleted);
            Assert.Null(observer.Finished[1].Usage);
        }
    }

    [Fact]
    public async Task ObserverRecordsUsageBeforeTypedResultDeserializationFails()
    {
        var observer = new RecordingObserver();
        var client = CreateClient(new MockHttpHandler(_ => Task.FromResult(BuildFinalResponse("invalid JSON"))));

        await Assert.ThrowsAsync<JsonException>(() => client.CompleteWithToolsAsync(
            ObservedRequest(observer), _testTools,
            _ => throw new InvalidOperationException("No tools expected"), ToolLoopJsonContext.Default.WeatherResult));

        var result = Assert.Single(observer.Finished);
        Assert.True(result.ResponseCompleted);
        Assert.Equal(15, result.Usage!.Value.TotalTokens);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task StreamObserverDistinguishesCompletedResponseFromEarlyDisposal(bool disposeEarly)
    {
        var observer = new RecordingObserver();
        var sse = "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n"
            + "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
            + "data: [DONE]\n\n";
        var client = CreateClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sse, System.Text.Encoding.UTF8, "text/event-stream")
        })));

        await foreach (var item in client.StreamWithToolsAsync(ObservedRequest(observer), _testTools,
            _ => throw new InvalidOperationException("No tools expected")))
        {
            if (disposeEarly) break;
            if (item is CompletedEvent) Assert.Single(observer.Finished);
        }

        var result = Assert.Single(observer.Finished);
        Assert.Equal(!disposeEarly, result.ResponseCompleted);
        Assert.Equal(7, result.Usage!.Value.TotalTokens);
    }

    [Fact]
    public async Task StreamObserverDeniesNextCallAfterToolUsageIsRecorded()
    {
        var observer = new RecordingObserver { DenyIteration = 2 };
        var calls = 0;
        var sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]}}]}\n\n"
            + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n"
            + "data: [DONE]\n\n";
        var client = CreateClient(new MockHttpHandler(_ =>
        {
            calls++;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sse, System.Text.Encoding.UTF8, "text/event-stream")
            });
        }));

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var item in client.StreamWithToolsAsync(ObservedRequest(observer), _testTools, call =>
            {
                Assert.Single(observer.Finished);
                return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "sunny" });
            })) { }
        });

        Assert.Equal(1, calls);
        Assert.Equal(2, observer.Started.Count);
        var result = Assert.Single(observer.Finished);
        Assert.True(result.ResponseCompleted);
        Assert.Equal(7, result.Usage!.Value.TotalTokens);
        Assert.Equal("sunny", observer.Started[1].Request.Messages[^1].Content);
    }

    [Fact]
    public async Task StreamObserverReportsKnownUsageOnCancellation()
    {
        var observer = new RecordingObserver();
        using var cancellation = new CancellationTokenSource();
        var sse = "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n"
            + "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
            + "data: [DONE]\n\n";
        var client = CreateClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sse, System.Text.Encoding.UTF8, "text/event-stream")
        })));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var item in client.StreamWithToolsAsync(ObservedRequest(observer), _testTools,
                _ => throw new InvalidOperationException("No tools expected"), ct: cancellation.Token))
            {
                cancellation.Cancel();
            }
        });

        var result = Assert.Single(observer.Finished);
        Assert.False(result.ResponseCompleted);
        Assert.Equal(7, result.Usage!.Value.TotalTokens);
    }

    private static CompletionRequest ObservedRequest(IToolIterationObserver observer) => new()
    {
        Model = AiModel.Gpt52,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather?" }],
        ToolIterationObserver = observer,
    };

    private sealed class RecordingObserver : IToolIterationObserver
    {
        public int? DenyIteration { get; init; }
        public List<ToolIterationRequest> Started { get; } = [];
        public List<ToolIterationResult> Finished { get; } = [];

        public ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken)
        {
            Started.Add(request);
            if (request.Iteration == DenyIteration) throw new InvalidOperationException("budget denied");
            return ValueTask.CompletedTask;
        }

        public ValueTask OnFinishedAsync(ToolIterationResult result)
        {
            Finished.Add(result);
            return ValueTask.CompletedTask;
        }
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

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(ToolLoopTests.WeatherResult))]
internal sealed partial class ToolLoopJsonContext : JsonSerializerContext
{
}
