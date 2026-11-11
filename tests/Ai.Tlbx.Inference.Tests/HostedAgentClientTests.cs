using System.Net;
using System.Text;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;

namespace Ai.Tlbx.Inference.Tests;

public sealed class HostedAgentClientTests
{
    [Fact]
    public async Task CreateSession_UsesHostedEnvironmentAndServiceMcp()
    {
        var handler = new FakeHandler((request, body) =>
        {
            Assert.Equal(HttpMethod.Post, request.Method);
            Assert.Equal("agents=v1", request.Headers.GetValues("OpenAI-Beta").Single());
            Assert.Equal("Bearer test-key", request.Headers.GetValues("Authorization").Single());
            using var json = JsonDocument.Parse(body!);
            var root = json.RootElement;
            Assert.Equal("gpt-6-astra", root.GetProperty("agent").GetProperty("model").GetString());
            Assert.Equal("openai_hosted", root.GetProperty("environment").GetProperty("type").GetString());
            Assert.Equal("disabled", root.GetProperty("environment").GetProperty("network").GetProperty("access").GetString());
            var mcp = root.GetProperty("agent").GetProperty("tools")[1];
            Assert.Equal("service", mcp.GetProperty("connection_origin").GetString());
            Assert.Equal("http", mcp.GetProperty("transport").GetProperty("type").GetString());
            Assert.Equal("search", mcp.GetProperty("allowed_tools")[0].GetString());
            return Json("{\"id\":\"sess_1\",\"status\":\"idle\"}");
        });
        var client = Client(handler);
        var session = await client.CreateHostedAgentSessionAsync(new HostedAgentOptions
        {
            Model = "gpt-6-astra", EnableWebSearch = true,
            McpServers = [new HostedAgentMcpServer { Label = "docs", Url = new Uri("https://example.com/mcp"), AllowedTools = ["search"] }]
        });
        Assert.Equal("sess_1", session.Id);
    }

    [Fact]
    public async Task Turn_SubscribesBeforeSending_AndStopsOnlyOnRootCompletion()
    {
        var calls = new List<string>();
        var handler = new FakeHandler((request, body) =>
        {
            calls.Add(request.Method + " " + request.RequestUri!.AbsolutePath);
            if (request.Method == HttpMethod.Get)
            {
                var sse = "data: {\"type\":\"agent.session.turn.output_text.delta\",\"delta\":\"hello\"}\n\n" +
                          "data: {\"type\":\"agent.session.turn.completed\",\"turn\":{\"id\":\"sub\",\"subagent_id\":\"a\"}}\n\n" +
                          "data: {\"type\":\"agent.session.turn.completed\",\"turn\":{\"id\":\"root\",\"subagent_id\":null}}\n\n";
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(sse, Encoding.UTF8, "text/event-stream") };
            }
            using var json = JsonDocument.Parse(body!);
            Assert.Equal("agent.session.input.message", json.RootElement.GetProperty("events")[0].GetProperty("type").GetString());
            Assert.Equal("hi", json.RootElement.GetProperty("events")[0].GetProperty("input")[0].GetProperty("content")[0].GetProperty("text").GetString());
            return Json("{}");
        });
        var events = new List<HostedAgentEvent>();
        await foreach (var item in Client(handler).RunHostedAgentTurnAsync("sess_1", "hi")) events.Add(item);
        Assert.Equal(["GET /v1/agents/sessions/sess_1/events", "POST /v1/agents/sessions/sess_1/events"], calls);
        Assert.Equal(3, events.Count);
        Assert.Equal("hello", events[0].Delta);
        Assert.False(events[1].IsRootTurn);
        Assert.True(events[2].IsRootTurn);
    }

    [Fact]
    public async Task Turn_WhenStreamEnds_DoesNotResendTask()
    {
        var posts = 0;
        var handler = new FakeHandler((request, _) =>
        {
            if (request.Method == HttpMethod.Post) posts++;
            return request.Method == HttpMethod.Get
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("data: {\"type\":\"agent.session.idle\"}\n\n") }
                : Json("{}");
        });
        await Assert.ThrowsAsync<IOException>(async () =>
        {
            await foreach (var _ in Client(handler).RunHostedAgentTurnAsync("sess_1", "hi")) { }
        });
        Assert.Equal(1, posts);
    }

    [Fact]
    public async Task ListTurns_CollectsEveryPageAndParsesReportedUsage()
    {
        var calls = 0;
        var handler = new FakeHandler((request, _) =>
        {
            calls++;
            if (calls == 1)
            {
                Assert.DoesNotContain("after=", request.RequestUri!.Query);
                return Json("{\"data\":[{\"id\":\"turn_1\",\"status\":\"completed\",\"subagent_id\":null,\"usage\":{\"input_tokens\":500,\"input_tokens_details\":{\"cached_tokens\":100},\"output_tokens\":20,\"output_tokens_details\":{\"reasoning_tokens\":5}}}],\"has_more\":true,\"last_id\":\"turn_1\"}");
            }
            Assert.Contains("after=turn_1", request.RequestUri!.Query);
            return Json("{\"data\":[{\"id\":\"turn_2\",\"status\":\"completed\",\"subagent_id\":\"sub_1\",\"usage\":null}],\"has_more\":false}");
        });
        var turns = await Client(handler).ListHostedAgentTurnsAsync("sess_1");
        Assert.Equal(2, calls);
        Assert.Equal(2, turns.Count);
        Assert.Equal(500, turns[0].Usage!.Value.InputTokens);
        Assert.Equal(100, turns[0].Usage!.Value.CacheReadTokens);
        Assert.Equal(5, turns[0].Usage!.Value.ThinkingTokens);
        Assert.Equal("sub_1", turns[1].SubagentId);
        Assert.Null(turns[1].Usage);
    }

    [Theory]
    [InlineData("http://example.com/mcp")]
    [InlineData("/relative")]
    public async Task CreateSession_RejectsNonHttpsOrRelativeMcpBeforeHttp(string url)
    {
        var handler = new FakeHandler((_, _) => throw new InvalidOperationException("No HTTP expected"));
        await Assert.ThrowsAsync<ArgumentException>(() => Client(handler).CreateHostedAgentSessionAsync(new HostedAgentOptions
        {
            Model = "gpt-6-astra",
            McpServers = [new HostedAgentMcpServer { Label = "test", Url = new Uri(url, UriKind.RelativeOrAbsolute) }]
        }));
    }

    [Theory]
    [InlineData("agent.session.turn.failed")]
    [InlineData("agent.session.turn.cancelled")]
    [InlineData("agent.session.requires_action")]
    public async Task Turn_RejectsUnsuccessfulOrUnsupportedOutcomes(string type)
    {
        var handler = new FakeHandler((request, _) => request.Method == HttpMethod.Get
            ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(
                "data: {\"type\":\"" + type + "\",\"turn\":{\"id\":\"root\",\"subagent_id\":null}}\n\n") }
            : Json("{}"));
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in Client(handler).RunHostedAgentTurnAsync("sess_1", "hi")) { }
        });
    }

    [Fact]
    public async Task ListItems_RejectsPaginationThatDoesNotAdvance()
    {
        var handler = new FakeHandler((_, _) => Json("{\"data\":[],\"has_more\":true,\"last_id\":\"same\"}"));
        await Assert.ThrowsAsync<InvalidDataException>(() => Client(handler).ListHostedAgentItemsAsync("sess_1"));
    }

    [Fact]
    public async Task HostedAgentEndpoints_ReportHttpFailureWithoutRetryingMutations()
    {
        var calls = 0;
        var handler = new FakeHandler((_, _) => { calls++; return new HttpResponseMessage(HttpStatusCode.Forbidden) { Content = new StringContent("denied") }; });
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => Client(handler).CancelHostedAgentTurnAsync("sess_1"));
        Assert.Equal(HttpStatusCode.Forbidden, error.StatusCode);
        Assert.Equal(1, calls);
    }

    private static AiInferenceClient Client(HttpMessageHandler handler) =>
        new(new HttpClient(handler), new AiInferenceOptions().AddOpenAi("test-key"));

    private static HttpResponseMessage Json(string value) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(value, Encoding.UTF8, "application/json")
    };

    private sealed class FakeHandler(Func<HttpRequestMessage, string?, HttpResponseMessage> respond) : HttpMessageHandler
    {
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return respond(request, body);
        }
    }
}
