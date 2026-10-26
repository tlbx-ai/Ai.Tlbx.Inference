using System.Net;
using System.Text;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class ResponsesMaintenanceTests
{
    private const string Reasoning = """[{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"opaque-ciphertext"}]""";
    private const string Completed = """{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":7,"output_tokens":3}}""";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Completion_ReplaysVisibleAssistantHistoryAndEncryptedReasoning(bool hasToolCalls)
    {
        string? body = null;
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            body = await request.Content!.ReadAsStringAsync();
            return JsonResponse(Completed);
        }));
        var provider = CreateProvider(http);

        await provider.CompleteAsync(new ProviderRequest
        {
            ModelApiName = "gpt-6-astra",
            PreferredEndpoint = ModelEndpointFamily.Responses,
            Messages =
            [
                new ChatMessage { Role = ChatRole.User, Content = "Remember the blue drawing." },
                new ChatMessage
                {
                    Role = ChatRole.Assistant,
                    Content = "I drew the blue house.",
                    ProviderContentJson = Reasoning,
                    ToolCalls = hasToolCalls
                        ? [new ToolCallRequest { Id = "call_1", ProviderItemId = "fc_1", Name = "draw", Arguments = "{}" }]
                        : null,
                },
                new ChatMessage { Role = ChatRole.User, Content = "What did you draw?" },
            ],
        }, CancellationToken.None);

        using var document = JsonDocument.Parse(body!);
        var input = document.RootElement.GetProperty("input");
        Assert.Equal("reasoning", input[1].GetProperty("type").GetString());
        Assert.Equal("opaque-ciphertext", input[1].GetProperty("encrypted_content").GetString());
        Assert.Equal("assistant", input[2].GetProperty("role").GetString());
        Assert.Equal("output_text", input[2].GetProperty("content")[0].GetProperty("type").GetString());
        Assert.Equal("I drew the blue house.", input[2].GetProperty("content")[0].GetProperty("text").GetString());
        if (hasToolCalls)
            Assert.Equal("fc_1", input[3].GetProperty("id").GetString());
        Assert.Contains(document.RootElement.GetProperty("include").EnumerateArray(), value => value.GetString() == "reasoning.encrypted_content");
    }

    [Fact]
    public async Task Completion_ExposesOpaqueReasoningAlongsideToolCalls()
    {
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(JsonResponse("""
            {"status":"completed","output":[
              {"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"opaque-ciphertext"},
              {"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
            ],"usage":{"input_tokens":7,"output_tokens":3}}
            """))));
        var result = await CreateProvider(http).CompleteAsync(Request(), CancellationToken.None);

        using var reasoning = JsonDocument.Parse(result.ProviderContentJson!);
        Assert.Single(reasoning.RootElement.EnumerateArray());
        Assert.Equal("opaque-ciphertext", reasoning.RootElement[0].GetProperty("encrypted_content").GetString());
        Assert.Single(result.ToolCalls!);
        Assert.Empty(result.Content);
    }

    [Theory]
    [InlineData("response.completed")]
    [InlineData("response.incomplete")]
    public async Task Stream_PreservesTerminalReasoningAndReportedUsage(string terminalType)
    {
        var sse = $$$$"""
            data: {"type":"{{{{terminalType}}}}","response":{"status":"completed","output":[{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"opaque-ciphertext"},{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":7,"output_tokens":3}}}

            """;
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sse, Encoding.UTF8, "text/event-stream"),
        })));
        var events = new List<ProviderStreamEvent>();
        await foreach (var item in CreateProvider(http).StreamAsync(Request(), CancellationToken.None))
            events.Add(item);

        Assert.Equal("done", string.Concat(events.Select(item => item.TextDelta)));
        var terminal = Assert.Single(events, item => item.Usage is not null);
        Assert.Equal(7, terminal.Usage!.Value.InputTokens);
        using var reasoning = JsonDocument.Parse(terminal.ProviderContentJson!);
        Assert.Equal("opaque-ciphertext", reasoning.RootElement[0].GetProperty("encrypted_content").GetString());
    }

    [Theory]
    [InlineData("response.failed")]
    [InlineData("error")]
    public async Task Stream_FailureEventThrowsInsteadOfReportingSuccess(string eventType)
    {
        var sse = $$$$"""
            data: {"type":"{{{{eventType}}}}","error":{"code":"server_error","message":"provider failed"}}

            """;
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sse, Encoding.UTF8, "text/event-stream"),
        })));

        var exception = await Assert.ThrowsAsync<HttpRequestException>(async () =>
        {
            await foreach (var _ in CreateProvider(http).StreamAsync(Request(), CancellationToken.None)) { }
        });
        Assert.Contains("server_error", exception.Message);
    }

    [Fact]
    public async Task Stream_FailureReportsAvailableUsageBeforeThrowing()
    {
        const string sse = """
            data: {"type":"response.failed","response":{"error":{"code":"server_error"},"usage":{"input_tokens":11,"output_tokens":5}}}

            """;
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sse, Encoding.UTF8, "text/event-stream"),
        })));
        var observed = new List<TokenUsage>();
        await Assert.ThrowsAsync<HttpRequestException>(async () =>
        {
            await foreach (var item in CreateProvider(http).StreamAsync(Request(), CancellationToken.None))
            {
                if (item.Usage is { } usage)
                    observed.Add(usage);
            }
        });
        Assert.Equal(11, Assert.Single(observed).InputTokens);
        Assert.Equal(5, observed[0].OutputTokens);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Completion_SendsResponsesStructuredOutputSchema(bool xai)
    {
        string? body = null;
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            body = await request.Content!.ReadAsStringAsync();
            return JsonResponse(Completed);
        }));
        var context = new ProviderRequestContext
        {
            HttpClient = http,
            BaseUrl = xai ? "https://api.x.ai" : "https://api.openai.com",
            ApiKey = "test",
        };
        IProvider provider = xai ? new XaiProvider(context) : new OpenAiProvider(context);
        await provider.CompleteAsync(Request() with
        {
            JsonSchema = """{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}""",
        }, CancellationToken.None);

        using var document = JsonDocument.Parse(body!);
        var format = document.RootElement.GetProperty("text").GetProperty("format");
        Assert.Equal("json_schema", format.GetProperty("type").GetString());
        Assert.True(format.GetProperty("strict").GetBoolean());
        Assert.Equal("integer", format.GetProperty("schema").GetProperty("properties").GetProperty("answer").GetProperty("type").GetString());
        Assert.False(document.RootElement.TryGetProperty("response_format", out _));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FacadeToolLoop_ReplaysReasoningBeforeFunctionCall(bool streaming)
    {
        var callCount = 0;
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            callCount++;
            using var document = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            if (callCount == 2)
            {
                var input = document.RootElement.GetProperty("input");
                Assert.Equal("reasoning", input[1].GetProperty("type").GetString());
                Assert.Equal("opaque-ciphertext", input[1].GetProperty("encrypted_content").GetString());
                Assert.Equal("function_call", input[2].GetProperty("type").GetString());
                Assert.Equal("function_call_output", input[3].GetProperty("type").GetString());
            }

            const string toolOutput = """
                {"status":"completed","output":[
                  {"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"opaque-ciphertext"},
                  {"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
                ],"usage":{"input_tokens":7,"output_tokens":3}}
                """;
            var responseJson = callCount == 1 ? toolOutput : Completed;
            if (!streaming)
                return JsonResponse(responseJson);

            var sse = callCount == 1
                ? "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\"}}\n\n"
                : "";
            sse += "data: {\"type\":\"response.completed\",\"response\":" + responseJson.Replace("\r", "").Replace("\n", "") + "}\n\n";
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sse, Encoding.UTF8, "text/event-stream"),
            };
        }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenAi("test"));
        using var parameters = JsonDocument.Parse("""{"type":"object","properties":{},"additionalProperties":false}""");
        ToolDefinition[] tools = [new() { Name = "lookup", Description = "Look up the result.", ParametersSchema = parameters.RootElement }];
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt6Luna,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Look it up." }],
        };
        static Task<ToolCallResult> Execute(ToolCallRequest call) => Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "42" });
        if (streaming)
        {
            await foreach (var _ in client.StreamWithToolsAsync(request, tools, Execute)) { }
        }
        else
        {
            var response = await client.CompleteWithToolsAsync(request, tools, Execute);
            Assert.Equal("done", response.Content);
        }
        Assert.Equal(2, callCount);
    }

    private static OpenAiProvider CreateProvider(HttpClient http) => new(new ProviderRequestContext
    {
        HttpClient = http,
        BaseUrl = "https://api.openai.com",
        ApiKey = "test",
    });

    private static ProviderRequest Request() => new()
    {
        ModelApiName = "gpt-6-astra",
        PreferredEndpoint = ModelEndpointFamily.Responses,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Run lookup." }],
    };

    private static HttpResponseMessage JsonResponse(string json) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(json, Encoding.UTF8, "application/json"),
    };
}
