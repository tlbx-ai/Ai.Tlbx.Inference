using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class AnthropicMaintenanceTests
{
    [Theory]
    [InlineData("claude-fable-5-1")]
    [InlineData("claude-opus-5-5")]
    [InlineData("claude-sonnet-5-5")]
    [InlineData("claude-haiku-5-5")]
    public async Task CurrentModels_MapManualBudgetToAdaptiveEffortWithoutRaisingOutputLimit(string model)
    {
        using var payload = await CaptureRequestAsync(Request(model) with { ThinkingBudget = 16000, MaxTokens = 2048 });
        Assert.Equal("adaptive", payload.RootElement.GetProperty("thinking").GetProperty("type").GetString());
        Assert.False(payload.RootElement.GetProperty("thinking").TryGetProperty("budget_tokens", out _));
        Assert.Equal("high", payload.RootElement.GetProperty("output_config").GetProperty("effort").GetString());
        Assert.Equal(2048, payload.RootElement.GetProperty("max_tokens").GetInt32());
    }

    [Fact]
    public async Task ExplicitEffort_WinsOverBudget_AndCoexistsWithNativeJsonAndTools()
    {
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}""");
        using var payload = await CaptureRequestAsync(Request("claude-opus-5-5") with
        {
            ThinkingBudget = 16000,
            ReasoningEffort = "low",
            JsonSchema = schema.RootElement.GetRawText(),
            Tools = [new ToolDefinition { Name = "lookup", Description = "Lookup", ParametersSchema = schema.RootElement }],
        });
        var body = payload.RootElement;
        Assert.Equal("low", body.GetProperty("output_config").GetProperty("effort").GetString());
        Assert.Equal("json_schema", body.GetProperty("output_config").GetProperty("format").GetProperty("type").GetString());
        Assert.True(body.GetProperty("output_config").GetProperty("format").GetProperty("schema").GetProperty("properties").TryGetProperty("ok", out _));
        Assert.Equal("lookup", Assert.Single(body.GetProperty("tools").EnumerateArray()).GetProperty("name").GetString());
        Assert.False(body.TryGetProperty("tool_choice", out _));
    }

    [Theory]
    [InlineData("claude-opus-4-6")]
    [InlineData("claude-sonnet-4-6")]
    [InlineData("claude-haiku-4-5-20251001")]
    public async Task ExistingModels_UseNativeJsonWithoutSyntheticForcedTool(string model)
    {
        using var payload = await CaptureRequestAsync(Request(model) with { JsonSchema = """{"type":"object","properties":{},"additionalProperties":false}""" });
        Assert.True(payload.RootElement.GetProperty("output_config").TryGetProperty("format", out _));
        Assert.False(payload.RootElement.TryGetProperty("tools", out _));
        Assert.False(payload.RootElement.TryGetProperty("tool_choice", out _));
    }

    [Theory]
    [InlineData("claude-sonnet-5-5", "between_tools")]
    [InlineData("claude-haiku-5-5", "disabled")]
    public async Task NoneEffort_UsesModelsLowestThinkingSetting(string model, string thinkingType)
    {
        using var payload = await CaptureRequestAsync(Request(model) with { ReasoningEffort = "none" });
        Assert.Equal(thinkingType, payload.RootElement.GetProperty("thinking").GetProperty("type").GetString());
        Assert.Equal("low", payload.RootElement.GetProperty("output_config").GetProperty("effort").GetString());
    }

    [Theory]
    [InlineData("claude-fable-5-1", "none")]
    [InlineData("claude-opus-5-5", "none")]
    [InlineData("claude-sonnet-5-5", "ultra")]
    public async Task InvalidEffort_IsRejectedBeforeHttp(string model, string effort)
    {
        var provider = CreateProvider(_ => throw new InvalidOperationException("Must not call provider"));
        await Assert.ThrowsAsync<ArgumentException>(() => provider.CompleteAsync(Request(model) with { ReasoningEffort = effort }, CancellationToken.None));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task CurrentModels_RejectUnsupportedSamplingBeforeHttp(bool temperature)
    {
        var provider = CreateProvider(_ => throw new InvalidOperationException("Must not call provider"));
        await Assert.ThrowsAsync<ArgumentException>(() => provider.CompleteAsync(Request("claude-opus-5-5") with
        {
            Temperature = temperature ? 0.5 : null,
            TopP = temperature ? null : 0.5,
        }, CancellationToken.None));
    }

    [Fact]
    public async Task CompleteAsync_PreservesSignedThinkingAndToolBlocksForContinuation()
    {
        const string content = """[{"type":"thinking","thinking":"","signature":"opaque-signature"},{"type":"tool_use","id":"tool_1","name":"lookup","input":{"key":"value"}}]""";
        var provider = CreateProvider(_ => Task.FromResult(Response($$"""{"content":{{content}},"usage":{"input_tokens":10,"output_tokens":20},"stop_reason":"tool_use"}""")));
        var response = await provider.CompleteAsync(Request("claude-opus-5-5"), CancellationToken.None);
        Assert.Equal(content, response.ProviderContentJson);

        using var payload = await CaptureRequestAsync(Request("claude-opus-5-5") with
        {
            Messages = [new ChatMessage { Role = ChatRole.Assistant, ProviderContentJson = response.ProviderContentJson }],
        });
        var blocks = payload.RootElement.GetProperty("messages")[0].GetProperty("content");
        Assert.Equal("opaque-signature", blocks[0].GetProperty("signature").GetString());
        Assert.Equal("", blocks[0].GetProperty("thinking").GetString());
        Assert.Equal("value", blocks[1].GetProperty("input").GetProperty("key").GetString());
    }

    [Fact]
    public async Task StreamAsync_ReassemblesSignedThinkingAndToolArgumentsBeforeContinuation()
    {
        const string stream = """
            event: message_start
            data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}

            event: content_block_start
            data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Summary"}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed-"}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"state"}}

            event: content_block_stop
            data: {"type":"content_block_stop","index":0}

            event: content_block_start
            data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"key\":"}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"value\"}"}}

            event: content_block_stop
            data: {"type":"content_block_stop","index":1}

            event: message_delta
            data: {"type":"message_delta","usage":{"output_tokens":20},"delta":{"stop_reason":"tool_use"}}

            event: message_stop
            data: {"type":"message_stop"}

            """;
        var provider = CreateProvider(_ => Task.FromResult(Response(stream)));
        var events = new List<ProviderStreamEvent>();
        await foreach (var item in provider.StreamAsync(Request("claude-opus-5-5"), CancellationToken.None)) events.Add(item);
        Assert.Single(events, item => item.ToolCall is not null);
        var completed = Assert.Single(events, item => item.ProviderContentJson is not null);
        Assert.Equal(20, completed.Usage!.Value.OutputTokens);
        using var blocks = JsonDocument.Parse(completed.ProviderContentJson!);
        Assert.Equal("Summary", blocks.RootElement[0].GetProperty("thinking").GetString());
        Assert.Equal("signed-state", blocks.RootElement[0].GetProperty("signature").GetString());
        Assert.Equal("value", blocks.RootElement[1].GetProperty("input").GetProperty("key").GetString());
    }

    [Fact]
    public async Task StreamAsync_PreservesRedactedThinkingServerToolsAndCitations()
    {
        const string stream = """
            event: content_block_start
            data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque-encrypted-state"}}

            event: content_block_stop
            data: {"type":"content_block_stop","index":0}

            event: content_block_start
            data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"test\"}"}}

            event: content_block_stop
            data: {"type":"content_block_stop","index":1}

            event: content_block_start
            data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":"Initial ","citations":[]}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"answer"}}

            event: content_block_delta
            data: {"type":"content_block_delta","index":2,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://example.org","title":"Source","cited_text":"fact"}}}

            event: content_block_stop
            data: {"type":"content_block_stop","index":2}

            event: message_stop
            data: {"type":"message_stop"}

            """;
        var provider = CreateProvider(_ => Task.FromResult(Response(stream)));
        string? content = null;
        await foreach (var item in provider.StreamAsync(Request("claude-opus-5-5"), CancellationToken.None))
        {
            content = item.ProviderContentJson ?? content;
            Assert.Null(item.ToolCall);
        }
        using var blocks = JsonDocument.Parse(content!);
        Assert.Equal("opaque-encrypted-state", blocks.RootElement[0].GetProperty("data").GetString());
        Assert.Equal("test", blocks.RootElement[1].GetProperty("input").GetProperty("query").GetString());
        Assert.Equal("Initial answer", blocks.RootElement[2].GetProperty("text").GetString());
        Assert.Equal("fact", blocks.RootElement[2].GetProperty("citations")[0].GetProperty("cited_text").GetString());
    }

    private static ProviderRequest Request(string model) => new()
    {
        ModelApiName = model,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    };

    private static async Task<JsonDocument> CaptureRequestAsync(ProviderRequest request)
    {
        string? captured = null;
        var provider = CreateProvider(async message =>
        {
            captured = await message.Content!.ReadAsStringAsync();
            return Response("""{"content":[{"type":"text","text":"Hello"}],"usage":{"input_tokens":10,"output_tokens":5},"stop_reason":"end_turn"}""");
        });
        await provider.CompleteAsync(request, CancellationToken.None);
        return JsonDocument.Parse(captured!);
    }

    private static AnthropicProvider CreateProvider(Func<HttpRequestMessage, Task<HttpResponseMessage>> handler)
        => new(new ProviderRequestContext
        {
            HttpClient = new HttpClient(new MockHttpHandler(handler)),
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        });

    private static HttpResponseMessage Response(string content)
        => new(HttpStatusCode.OK) { Content = new StringContent(content, System.Text.Encoding.UTF8, "application/json") };
}
