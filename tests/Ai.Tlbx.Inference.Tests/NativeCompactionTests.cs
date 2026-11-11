using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests;

public sealed class NativeCompactionTests
{
    private const string Window = """[{"role":"developer","content":"rules"},{"role":"user","content":[{"type":"input_file","filename":"worksheet.pdf","file_data":"data:application/pdf;base64,JVBERg=="}]},{"type":"compaction","id":"cmp_1","encrypted_content":"opaque-state"}]""";
    private const string Usage = """{"input_tokens":12000,"output_tokens":2000,"input_tokens_details":{"cached_tokens":4000,"cache_write_tokens":7000,"cache_write_tokens_details":{"ttl_30m_tokens":3000,"ttl_12h_tokens":4000}}}""";
    private const string Final = """{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":7,"output_tokens":3}}""";
    private const string ToolResponse = """{"status":"completed","output":[{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"reasoning-cipher"},{"type":"message","content":[{"type":"output_text","text":"Let me check."}]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":7,"output_tokens":3}}""";
    private static readonly IReadOnlyList<ToolDefinition> Tools = [new()
    {
        Name = "lookup", Description = "Retrieve a document",
        ParametersSchema = JsonDocument.Parse("""{"type":"object","properties":{},"additionalProperties":false}""").RootElement.Clone()
    }];

    [Fact]
    public async Task Compact_UsesNativeSchemaAndPreservesEntireOpaqueWindowAndUsage()
    {
        var bodies = new List<JsonNode>();
        using var http = Http(async request =>
        {
            Assert.EndsWith("/responses/compact", request.RequestUri!.AbsolutePath);
            bodies.Add(JsonNode.Parse(await request.Content!.ReadAsStringAsync())!);
            return CompactResponse();
        });
        var result = await Client(http).CompactAsync(new CompactionRequest
        {
            Model = AiModel.Gpt61Terra, SystemMessage = "rules", Messages = [FileMessage()],
            ServiceTier = "fast", PromptCacheKey = "stable-prefix"
        });
        var body = Assert.Single(bodies);
        Assert.Equal("developer", body["input"]![0]!["role"]!.GetValue<string>());
        Assert.Equal("rules", body["input"]![0]!["content"]!.GetValue<string>());
        Assert.Equal("data:application/pdf;base64,JVBERg==", body["input"]![1]!["content"]![1]!["file_data"]!.GetValue<string>());
        foreach (var absent in new[] { "instructions", "tools", "reasoning", "stream" }) Assert.Null(body[absent]);
        Assert.Equal("fast", body["service_tier"]!.GetValue<string>());
        Assert.Equal("stable-prefix", body["prompt_cache_key"]!.GetValue<string>());
        Assert.Equal("cmp_response", result.Id);
        Assert.Equal(3, result.Messages.Count);
        var expected = JsonNode.Parse(Window)!.AsArray();
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.Equal(ChatRole.Provider, result.Messages[i].Role);
            Assert.True(JsonNode.DeepEquals(expected[i], JsonNode.Parse(result.Messages[i].ProviderContentJson!)));
        }
        Assert.Equal(12000, result.Usage.InputTokens);
        Assert.Equal(2000, result.Usage.OutputTokens);
        Assert.Equal(4000, result.Usage.CacheReadTokens);
        Assert.Equal(7000, result.Usage.CacheWriteTokens);
        Assert.Equal(3000, result.Usage.CacheWrite30MinuteTokens);
        Assert.Equal(4000, result.Usage.CacheWrite12HourTokens);
        Assert.Equal("fast", result.Usage.ServiceTier);
        Assert.Equal("fast", result.ServiceTier);
    }

    [Fact]
    public async Task CountInputTokens_IncludesDeveloperToolsInlineFilesAndReasoningEffort()
    {
        using var http = Http(async request =>
        {
            Assert.EndsWith("/responses/input_tokens", request.RequestUri!.AbsolutePath);
            var body = JsonNode.Parse(await request.Content!.ReadAsStringAsync())!;
            Assert.Equal("developer", body["input"]![0]!["role"]!.GetValue<string>());
            Assert.Equal("lookup", body["tools"]![0]!["name"]!.GetValue<string>());
            Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Tools[0].ParametersSchema.GetRawText()), body["tools"]![0]!["parameters"]));
            Assert.Equal("high", body["reasoning"]!["effort"]!.GetValue<string>());
            Assert.Equal("data:application/pdf;base64,JVBERg==", body["input"]![1]!["content"]![1]!["file_data"]!.GetValue<string>());
            return Json("""{"input_tokens":1234}""");
        });
        Assert.Equal(1234, await Client(http).CountInputTokensAsync(Request([FileMessage()]) with { ReasoningEffort = "high" }, Tools));
    }

    [Theory]
    [InlineData(239231, false, false)]
    [InlineData(239232, true, false)]
    [InlineData(239233, true, false)]
    [InlineData(267904, true, false)]
    [InlineData(267905, false, true)]
    [InlineData(272000, false, true)]
    [InlineData(272001, false, true)]
    public async Task DefaultTerraPolicy_RespectsTriggerAndCompactionBillingHeadroom(int tokens, bool compacts, bool fails)
    {
        var counts = 0; var compactCalls = 0; var modelCalls = 0;
        var observer = new Observer();
        var iterations = new IterationObserver();
        var original = new ChatMessage { Role = ChatRole.User, Content = "Keep every original byte." };
        using var http = Http(request => Task.FromResult(request.RequestUri!.AbsolutePath switch
        {
            var p when p.EndsWith("/input_tokens", StringComparison.Ordinal) => Json($"{{\"input_tokens\":{(++counts == 1 ? tokens : 10)}}}"),
            var p when p.EndsWith("/compact", StringComparison.Ordinal) => CountCompact(),
            _ => CountModel()
        }));
        HttpResponseMessage CountCompact() { compactCalls++; return CompactResponse(); }
        HttpResponseMessage CountModel() { modelCalls++; return Json(Final); }
        var request = Request([original]) with { Compaction = new(), CompactionObserver = observer, ToolIterationObserver = iterations };
        if (fails) await Assert.ThrowsAsync<InvalidOperationException>(() => Client(http).CompleteWithToolsAsync(request, Tools, Execute));
        else
        {
            var result = await Client(http).CompleteWithToolsAsync(request, Tools, Execute);
            Assert.Equal(original, Assert.Single(result.Messages));
            Assert.Equal(compacts ? 1 : 0, result.Compactions.Count);
        }
        Assert.Equal(compacts ? 1 : 0, compactCalls);
        Assert.Equal(fails ? 0 : 1, modelCalls);
        if (fails) Assert.Empty(iterations.Requests);
        else Assert.Equal(compacts ? 10 : tokens, Assert.Single(iterations.Requests).InputTokens);
        Assert.Equal(compacts ? tokens : (int?)null, observer.Started.SingleOrDefault()?.InputTokens);
        Assert.Same(original, Assert.Single(request.Messages));
    }

    [Fact]
    public async Task ToolLoop_RechecksAfterToolResultPreservesCanonicalTranscriptAndReplaysFullCheckpoint()
    {
        var paths = new List<string>(); var count = 0; var model = 0;
        var largeResult = new string('x', 80000);
        JsonNode? compactBody = null; JsonNode? finalBody = null;
        using var http = Http(async request =>
        {
            var path = request.RequestUri!.AbsolutePath;
            paths.Add(path.Split('/')[^1]);
            var body = JsonNode.Parse(await request.Content!.ReadAsStringAsync())!;
            if (path.EndsWith("/input_tokens", StringComparison.Ordinal)) return Json($"{{\"input_tokens\":{(++count == 2 ? 239232 : 10)}}}");
            if (path.EndsWith("/compact", StringComparison.Ordinal)) { compactBody = body; return CompactResponse(); }
            if (++model == 1) return Json(ToolResponse);
            finalBody = body; return Json(Final);
        });
        var original = new ChatMessage { Role = ChatRole.User, Content = "Original task" };
        var client = Client(http);
        var result = await client.CompleteWithToolsAsync(Request([original]) with { Compaction = new() }, Tools,
            call => Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = largeResult }));
        Assert.Equal(new[] { "input_tokens", "responses", "input_tokens", "compact", "input_tokens", "responses" }, paths);
        Assert.Equal(3, result.Messages.Count);
        Assert.Equal(original, result.Messages[0]);
        Assert.Equal("Let me check.", result.Messages[1].Content);
        Assert.Equal("fc_1", Assert.Single(result.Messages[1].ToolCalls!).ProviderItemId);
        Assert.Equal(largeResult, result.Messages[2].Content);
        Assert.Equal(largeResult, compactBody!["input"]!.AsArray().Last()!["output"]!.GetValue<string>());
        var state = Assert.IsType<CompactionState>(result.CompactionState);
        Assert.Equal(3, state.SourceMessageCount);
        Assert.Equal(3, state.Messages.Count);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Window), finalBody!["input"]));
        // Inline PDF bytes retained by compaction survive without the transient upload/delete lifecycle.
        Assert.DoesNotContain(paths, p => p == "files");

        JsonNode? nextBody = null;
        using var nextHttp = Http(async request => { nextBody = JsonNode.Parse(await request.Content!.ReadAsStringAsync()); return Json(Final); });
        await Client(nextHttp).CompleteWithToolsAsync(Request(result.Messages.Append(new ChatMessage { Role = ChatRole.User, Content = "Continue" }).ToArray()) with { CompactionState = state }, Tools, Execute);
        Assert.Equal(4, nextBody!["input"]!.AsArray().Count);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Window)![0], nextBody["input"]![0]));
        Assert.Equal("Continue", nextBody["input"]![3]!["content"]![0]!["text"]!.GetValue<string>());
        Assert.Single(nextBody["input"]!.AsArray(), item => item?["role"]?.GetValue<string>() == "developer");
    }

    [Fact]
    public async Task NonstreamObserver_RetainsFullAssistantBeforeFailingToolExecution()
    {
        var observer = new ConversationObserver();
        var requests = 0;
        using var http = Http(_ => { requests++; return Task.FromResult(Json(ToolResponse)); });
        var failure = new InvalidOperationException("Tool execution failed before another model iteration.");
        var caught = await Assert.ThrowsAsync<InvalidOperationException>(() => Client(http).CompleteWithToolsAsync(
            Request([new() { Role = ChatRole.User, Content = "Find the document" }]) with { ConversationObserver = observer },
            Tools, call =>
            {
                var persisted = Assert.Single(observer.Messages);
                Assert.Equal(ChatRole.Assistant, persisted.Role);
                Assert.Equal("Let me check.", persisted.Content);
                var persistedCall = Assert.Single(persisted.ToolCalls!);
                Assert.Equal(call, persistedCall);
                Assert.Equal("call_1", persistedCall.Id);
                Assert.Equal("fc_1", persistedCall.ProviderItemId);
                Assert.Equal("lookup", persistedCall.Name);
                Assert.Equal("{}", persistedCall.Arguments);
                Assert.Equal("reasoning-cipher", JsonNode.Parse(persisted.ProviderContentJson!)![0]!["encrypted_content"]!.GetValue<string>());
                throw failure;
            }));
        Assert.Same(failure, caught);
        Assert.Equal(1, requests);
        Assert.Single(observer.Messages);
        Assert.Equal("Let me check.", observer.Messages[0].Content);
        Assert.Equal("fc_1", observer.Messages[0].ToolCalls![0].ProviderItemId);
    }

    [Fact]
    public async Task Streaming_RecountsEveryToolIterationAndEmitsFullCompactionCheckpoint()
    {
        var counts = 0; var model = 0; var compactCalls = 0;
        var observer = new Observer();
        using var http = Http(request =>
        {
            var path = request.RequestUri!.AbsolutePath;
            if (path.EndsWith("/input_tokens", StringComparison.Ordinal)) return Task.FromResult(Json($"{{\"input_tokens\":{(++counts == 2 ? 239232 : 10)}}}"));
            if (path.EndsWith("/compact", StringComparison.Ordinal)) { compactCalls++; return Task.FromResult(CompactResponse()); }
            return Task.FromResult(Sse(++model == 1
                ? "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":" + ToolResponse + "}\n\n"
                : "data: {\"type\":\"response.completed\",\"response\":" + Final + "}\n\n"));
        });
        var events = new List<ToolStreamEvent>();
        await foreach (var item in Client(http).StreamWithToolsAsync(Request([new() { Role = ChatRole.User, Content = "task" }]) with { Compaction = new(), CompactionObserver = observer }, Tools, Execute)) events.Add(item);
        Assert.Equal(3, counts); Assert.Equal(2, model); Assert.Equal(1, compactCalls);
        var compacted = Assert.Single(events.OfType<ConversationCompactedEvent>());
        Assert.Equal(3, compacted.Compaction.State.SourceMessageCount);
        Assert.Equal(3, compacted.Compaction.State.Messages.Count);
        Assert.Equal(3, compacted.Compaction.SourceMessages.Count);
        Assert.Equal(ChatRole.Tool, compacted.Compaction.SourceMessages[2].Role);
        Assert.Equal(compacted.Compaction.State, Assert.Single(events.OfType<CompletedEvent>()).CompactionState);
        Assert.Single(observer.Completed);
    }

    [Fact]
    public async Task Streaming_JournalsFullAssistantBeforeExecutionAndExactToolResultBeforeLegacyEvent()
    {
        var model = 0; ChatMessage? assistant = null; ChatMessage? tool = null;
        var legacyResultSeen = false;
        using var http = Http(_ => Task.FromResult(++model == 1
            ? Sse("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Let me check.\"}\n\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":" + ToolResponse + "}\n\n")
            : Sse("data: {\"type\":\"response.completed\",\"response\":" + Final + "}\n\n")));
        await foreach (var item in Client(http).StreamWithToolsAsync(Request([new() { Role = ChatRole.User, Content = "Find it" }]), Tools, call =>
        {
            Assert.NotNull(assistant);
            Assert.Equal("Let me check.", assistant.Content);
            Assert.Equal("reasoning-cipher", JsonNode.Parse(assistant.ProviderContentJson!)![0]!["encrypted_content"]!.GetValue<string>());
            Assert.Equal("call_1", call.Id); Assert.Equal("fc_1", call.ProviderItemId);
            Assert.Equal("{}", call.Arguments);
            return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "exact tool output" });
        }))
        {
            if (item is ConversationMessageAddedEvent added && added.Message.Role == ChatRole.Assistant && added.Message.ToolCalls is not null) assistant = added.Message;
            if (item is ConversationMessageAddedEvent addedTool && addedTool.Message.Role == ChatRole.Tool) tool = addedTool.Message;
            if (item is ToolResultAddedEvent legacy)
            {
                Assert.NotNull(tool); Assert.Equal("call_1", tool.ToolCallId);
                Assert.Equal(tool.Content, legacy.Result); Assert.Equal(tool.ToolCallId, legacy.ToolCallId);
                legacyResultSeen = true;
            }
        }
        Assert.True(legacyResultSeen);
    }

    [Theory]
    [InlineData("compact", 0, 1)]
    [InlineData("count-after", 1, 0)]
    [InlineData("model", 1, 0)]
    [InlineData("too-large", 1, 0)]
    public async Task Failures_PreserveCompletedCompactionAccountingAndNeverGenerateWithUnverifiedBudget(string failure, int completed, int failed)
    {
        var counts = 0; var model = 0; var observer = new Observer();
        using var http = Http(request =>
        {
            var path = request.RequestUri!.AbsolutePath;
            if (path.EndsWith("/input_tokens", StringComparison.Ordinal))
            {
                if (++counts == 1) return Task.FromResult(Json("""{"input_tokens":239232}"""));
                return Task.FromResult(failure == "count-after" ? Error() : Json($"{{\"input_tokens\":{(failure == "too-large" ? 239232 : 10)}}}"));
            }
            if (path.EndsWith("/compact", StringComparison.Ordinal)) return Task.FromResult(failure == "compact" ? Error() : CompactResponse());
            model++; return Task.FromResult(Error());
        });
        await Assert.ThrowsAnyAsync<Exception>(() => Client(http).CompleteWithToolsAsync(Request([new() { Role = ChatRole.User, Content = "task" }]) with { Compaction = new(), CompactionObserver = observer }, Tools, Execute));
        Assert.Single(observer.Started); Assert.Equal(completed, observer.Completed.Count); Assert.Equal(failed, observer.Failed.Count);
        Assert.Equal(failure == "model" ? 1 : 0, model);
        if (completed == 1)
        {
            Assert.Equal(12000, observer.Completed[0].Response.Usage.InputTokens);
            Assert.Equal(failure == "count-after" ? -1 : failure == "too-large" ? 239232 : 10, observer.Completed[0].CompactedInputTokens);
        }
    }

    [Theory]
    [InlineData(AiModel.Gemini38Flash)]
    [InlineData(AiModel.ClaudeSonnet55)]
    public async Task UnsupportedProviders_RejectNativeContextOperationsBeforeHttp(AiModel model)
    {
        using var http = Http(_ => throw new InvalidOperationException("No HTTP request is permitted."));
        var client = Client(http);
        await Assert.ThrowsAsync<NotSupportedException>(() => client.CompactAsync(new CompactionRequest { Model = model, Messages = [] }));
        await Assert.ThrowsAsync<NotSupportedException>(() => client.CountInputTokensAsync(Request([]) with { Model = model }));
    }

    [Fact]
    public void CacheWriteLifetimeUsage_AggregatesWithoutDoubleCountingAndUsesExactPricingBoundary()
    {
        var usage = new TokenUsage { InputTokens = 12000, OutputTokens = 2000, CacheReadTokens = 4000, CacheWriteTokens = 7000, CacheWrite30MinuteTokens = 3000, CacheWrite12HourTokens = 4000 };
        var rates = AiModelCostCatalog.GetRates(AiModel.Gpt61Terra);
        Assert.Equal(272001, rates.LongContextStartsAtTokens);
        Assert.Equal(1.25m, rates.CacheWriteInputPerMillion); Assert.Equal(2.5m, rates.CacheWrite12HourInputPerMillion);
        Assert.Equal(.02495m, TokenCostCalculator.Estimate(usage, rates).ProviderCost);
        Assert.Equal(.564902m, TokenCostCalculator.Estimate(usage with { InputTokens = 272001 }, rates).ProviderCost);
        var sum = usage + usage;
        Assert.Equal(6000, sum.CacheWrite30MinuteTokens); Assert.Equal(8000, sum.CacheWrite12HourTokens); Assert.Equal(14000, sum.CacheWriteTokens);
    }

    private static CompletionRequest Request(IReadOnlyList<ChatMessage> messages) => new() { Model = AiModel.Gpt61Terra, SystemMessage = "rules", Messages = messages };
    private static ChatMessage FileMessage() => new() { Role = ChatRole.User, Content = "Inspect this", Attachments = [new() { FileName = "worksheet.pdf", MimeType = "application/pdf", Content = Encoding.ASCII.GetBytes("%PDF") }] };
    private static AiInferenceClient Client(HttpClient http) => new(http, new AiInferenceOptions().AddOpenAi("test"));
    private static HttpClient Http(Func<HttpRequestMessage, Task<HttpResponseMessage>> handler) => new(new MockHttpHandler(handler));
    private static HttpResponseMessage Json(string body) => new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    private static HttpResponseMessage Sse(string body) => new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "text/event-stream") };
    private static HttpResponseMessage Error() => new(HttpStatusCode.BadRequest) { Content = new StringContent("{\"error\":{\"message\":\"fixture failure\"}}") };
    private static HttpResponseMessage CompactResponse() => Json("{\"id\":\"cmp_response\",\"object\":\"response.compaction\",\"service_tier\":\"fast\",\"output\":" + Window + ",\"usage\":" + Usage + "}");
    private static Task<ToolCallResult> Execute(ToolCallRequest call) => Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = "result" });
    private sealed class IterationObserver : IToolIterationObserver
    {
        public List<ToolIterationRequest> Requests { get; } = [];
        public ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken) { Requests.Add(request); return ValueTask.CompletedTask; }
        public ValueTask OnFinishedAsync(ToolIterationResult result) => ValueTask.CompletedTask;
    }

    private sealed class ConversationObserver : IConversationObserver
    {
        public List<ChatMessage> Messages { get; } = [];
        public ValueTask OnMessageAsync(ChatMessage message, CancellationToken cancellationToken)
        {
            Messages.Add(message);
            return ValueTask.CompletedTask;
        }
    }

    private sealed class Observer : ICompactionObserver
    {
        public List<CompactionRequest> Started { get; } = [];
        public List<ConversationCompaction> Completed { get; } = [];
        public List<Exception> Failed { get; } = [];
        public ValueTask OnStartingAsync(CompactionRequest request, CancellationToken ct) { Started.Add(request); return ValueTask.CompletedTask; }
        public ValueTask OnCompletedAsync(ConversationCompaction result, CancellationToken ct) { Completed.Add(result); return ValueTask.CompletedTask; }
        public ValueTask OnFailedAsync(Exception error, CancellationToken ct) { Failed.Add(error); return ValueTask.CompletedTask; }
    }
}
