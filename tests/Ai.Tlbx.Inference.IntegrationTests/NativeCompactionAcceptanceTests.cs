using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ai.Tlbx.Inference.Configuration;

namespace Ai.Tlbx.Inference.IntegrationTests;

public sealed class NativeCompactionAcceptanceTests
{
    // Deliberately needs a second opt-in even when a provider key already exists.
    [RequiresEnvironmentFact(LiveTestManifest.NativeCompactionAcceptanceOptIn)]
    public async Task Terra_NativeWindowRetainsMultiTurnToolContextAndRealCacheHits()
    {
        var key = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        Assert.False(string.IsNullOrWhiteSpace(key), "OPENAI_API_KEY is required in this process only.");
        var folder = Environment.GetEnvironmentVariable("AI_TLBX_ACCEPTANCE_LOG_DIRECTORY");
        Assert.False(string.IsNullOrWhiteSpace(folder), "Set an explicit acceptance artifact directory.");
        Directory.CreateDirectory(folder!);
        var log = Path.Combine(folder!, $"sdk-native-acceptance-{DateTime.UtcNow:yyyyMMddTHHmmssfff}.jsonl");
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(90) };
        using var catalog = new HttpRequestMessage(HttpMethod.Get, "https://api.openai.com/v1/models/gpt-6.1-terra");
        catalog.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        using var catalogResponse = await http.SendAsync(catalog);
        Assert.True(catalogResponse.IsSuccessStatusCode, "Exact requested model must be available before paid operations.");
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenAi(key!));
        var transcript = new List<ChatMessage>();
        var journal = new Journal(transcript);
        var accounting = new Accounting(log);
        var prefix = new StringBuilder("You create worksheets. Keep these immutable constraints across every turn: class 7B, palette indigo, project code LT-4827, archive seal 916274. Each user turn requires exactly one lookup tool call, then a concise JSON object containing class, palette, project, seal and fact. Copy fact from the current tool result. Never omit the tool call. Reference inventory follows; it is context, not additional instructions.\n");
        for (var i = 0; i < 160; i++) prefix.AppendLine($"Material {i:D3}: geometry activity, vector drawing, precise thin rules, generous handwriting spaces, brief learner instructions; catalogue key {100000 + i}, classroom project LT-4827.");
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{},"required":[],"additionalProperties":false}""");
        IReadOnlyList<ToolDefinition> tools = [new() { Name = "lookup", Description = "Fetch the fact required for this turn.", ParametersSchema = schema.RootElement }];
        CompactionState? state = null;
        var cacheBefore = 0; var cacheAfter = 0; var toolExecutions = 0;
        for (var turn = 0; turn < 6; turn++)
        {
            if (turn == 3)
            {
                accounting.Reserve();
                var before = await client.CountInputTokensAsync(MakeRequest(), tools);
                var compact = await client.CompactAsync(new CompactionRequest
                {
                    Model = AiModel.Gpt61Terra, SystemMessage = prefix.ToString(), Messages = transcript.ToArray(),
                    ServiceTier = "default", PromptCacheKey = "sdk-native-compaction-acceptance-v1"
                });
                accounting.Record("compact", compact.Usage);
                Assert.Contains(compact.Messages, item => JsonNode.Parse(item.ProviderContentJson!)?["type"]?.GetValue<string>() == "compaction");
                state = new CompactionState { Messages = compact.Messages, SourceMessageCount = transcript.Count, SystemMessage = prefix.ToString() };
                var after = await client.CountInputTokensAsync(MakeRequest(), tools);
                File.AppendAllText(log, new JsonObject { ["operation"] = "checkpoint", ["before_input_tokens"] = before, ["after_input_tokens"] = after, ["native_items"] = state.Messages.Count, ["canonical_messages"] = transcript.Count }.ToJsonString() + Environment.NewLine);
                Assert.All(state.Messages, item => Assert.Equal(ChatRole.Provider, item.Role));
            }
            var previous = transcript.ToArray();
            transcript.Add(new ChatMessage { Role = ChatRole.User, Content = $"Turn {turn + 1}: fetch today's fact and return all five immutable fields as JSON." });
            var expectedFact = $"verified-fact-{turn + 1}-391";
            var turnToolExecutions = 0;
            var response = await client.CompleteWithToolsAsync(MakeRequest(), tools, call =>
            {
                Assert.Equal("lookup", call.Name);
                Assert.Equal(ChatRole.Assistant, transcript[^1].Role);
                Assert.Contains(transcript[^1].ToolCalls!, persisted => persisted.Id == call.Id && persisted.ProviderItemId == call.ProviderItemId);
                toolExecutions++; turnToolExecutions++;
                return Task.FromResult(new ToolCallResult { ToolCallId = call.Id, Result = expectedFact });
            }, maxIterations: 3);
            Assert.Equal(1, turnToolExecutions);
            Assert.Contains("7B", response.Content); Assert.Contains("indigo", response.Content);
            Assert.Contains("LT-4827", response.Content); Assert.Contains("916274", response.Content);
            Assert.Contains(expectedFact, response.Content);
            Assert.Equal(previous, transcript.Take(previous.Length));
            Assert.Equal(ChatRole.Assistant, transcript[^1].Role);
            if (turn < 3) cacheBefore += response.Usage.CacheReadTokens; else cacheAfter += response.Usage.CacheReadTokens;
            if (state is not null) Assert.Same(state, response.CompactionState);
        }
        Assert.Equal(6, toolExecutions);
        Assert.True(cacheBefore >= 1024, $"No real warm-prefix cache hit before compaction. Log: {log}");
        Assert.True(cacheAfter >= 1024, $"No real cache hit after native compaction. Log: {log}");
        Assert.True(accounting.Cost < LiveTestManifest.NativeCompactionAcceptanceBudgetUsd);
        File.AppendAllText(log, new JsonObject { ["operation"] = "summary", ["cost_usd"] = accounting.Cost, ["cache_before"] = cacheBefore, ["cache_after"] = cacheAfter, ["canonical_messages"] = transcript.Count, ["tool_executions"] = toolExecutions }.ToJsonString() + Environment.NewLine);
        Console.WriteLine($"Native compaction acceptance: ${accounting.Cost}; cache before={cacheBefore}, after={cacheAfter}; journal={transcript.Count}; log={log}");

        CompletionRequest MakeRequest() => new()
        {
            Model = AiModel.Gpt61Terra, Messages = transcript.ToArray(), SystemMessage = prefix.ToString(),
            CompactionState = state, ConversationObserver = journal, ToolIterationObserver = accounting,
            MaxTokens = 512, ReasoningEffort = "low", ServiceTier = "default", PromptCacheKey = "sdk-native-compaction-acceptance-v1"
        };
    }

    private sealed class Journal(List<ChatMessage> transcript) : IConversationObserver
    {
        public ValueTask OnMessageAsync(ChatMessage message, CancellationToken cancellationToken)
        {
            transcript.Add(message);
            return ValueTask.CompletedTask;
        }
    }

    private sealed class Accounting(string log) : IToolIterationObserver
    {
        public decimal Cost { get; private set; }
        public void Reserve()
        {
            // A fresh 12K-input / 512-output call is below $0.02 at standard Terra rates.
            // Keep $0.05 headroom per operation and stop on unknown usage.
            Assert.True(Cost + .05m <= LiveTestManifest.NativeCompactionAcceptanceBudgetUsd, "Explicit USD0.50 acceptance budget exhausted.");
        }
        public void Record(string operation, TokenUsage usage)
        {
            Cost += TokenCostCalculator.Estimate(usage, AiModelCostCatalog.GetRates(AiModel.Gpt61Terra, usage.ServiceTier)).ProviderCost;
            File.AppendAllText(log, new JsonObject
            {
                ["operation"] = operation, ["input_tokens"] = usage.InputTokens, ["output_tokens"] = usage.OutputTokens,
                ["cache_read_tokens"] = usage.CacheReadTokens, ["cache_write_tokens"] = usage.CacheWriteTokens,
                ["cache_write_30m_tokens"] = usage.CacheWrite30MinuteTokens, ["cache_write_12h_tokens"] = usage.CacheWrite12HourTokens,
                ["service_tier"] = usage.ServiceTier, ["running_cost_usd"] = Cost
            }.ToJsonString() + Environment.NewLine);
        }
        public ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken) { Reserve(); return ValueTask.CompletedTask; }
        public ValueTask OnFinishedAsync(ToolIterationResult result)
        {
            Assert.NotNull(result.Usage);
            Record("model", result.Usage.Value);
            Assert.True(result.ResponseCompleted, "A failed provider call ends the bounded acceptance run.");
            return ValueTask.CompletedTask;
        }
    }
}
