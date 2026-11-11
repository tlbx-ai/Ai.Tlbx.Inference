using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ai.Tlbx.Inference;

/// <summary>Direct HTTP adapter for the beta Agents API. OpenAI owns the hosted tool loop.</summary>
internal sealed class HostedAgentClient
{
    private const string SessionsUrl = "https://api.openai.com/v1/agents/sessions";
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;

    internal HostedAgentClient(HttpClient httpClient, string apiKey)
    {
        _httpClient = httpClient;
        _apiKey = apiKey;
    }

    internal async Task<HostedAgentSession> CreateAsync(HostedAgentOptions options, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrWhiteSpace(options.Model))
            throw new ArgumentException("An agent model is required.", nameof(options));

        var agent = new JsonObject { ["model"] = options.Model };
        if (!string.IsNullOrWhiteSpace(options.Instructions)) agent["instructions"] = options.Instructions;
        if (!string.IsNullOrWhiteSpace(options.ReasoningEffort))
            agent["reasoning"] = new JsonObject { ["effort"] = options.ReasoningEffort };

        var tools = new JsonArray();
        if (options.EnableWebSearch)
            tools.Add((JsonNode)new JsonObject { ["type"] = "web_search", ["mode"] = "live" });
        foreach (var server in options.McpServers)
        {
            if (string.IsNullOrWhiteSpace(server.Label) || server.Url is null || !server.Url.IsAbsoluteUri || server.Url.Scheme != Uri.UriSchemeHttps)
                throw new ArgumentException("MCP servers require a label and an HTTPS URL.", nameof(options));
            var tool = new JsonObject
            {
                ["type"] = "mcp",
                ["server_label"] = server.Label,
                ["transport"] = new JsonObject { ["type"] = "http", ["server_url"] = server.Url.ToString() },
                ["connection_origin"] = "service",
                ["required"] = true
            };
            if (server.AllowedTools is { Count: > 0 })
            {
                var allowed = new JsonArray();
                foreach (var name in server.AllowedTools) allowed.Add((JsonNode?)JsonValue.Create(name));
                tool["allowed_tools"] = allowed;
            }
            tools.Add((JsonNode)tool);
        }
        if (tools.Count > 0) agent["tools"] = tools;

        var payload = new JsonObject
        {
            ["agent"] = agent,
            ["environment"] = new JsonObject
            {
                ["type"] = "openai_hosted",
                ["network"] = new JsonObject { ["access"] = options.AllowNetworkAccess ? "enabled" : "disabled" }
            }
        };
        // Create without input so callers can subscribe before the first turn.
        using var response = await SendAsync(HttpMethod.Post, SessionsUrl, payload, ct).ConfigureAwait(false);
        using var json = await ParseAsync(response, ct).ConfigureAwait(false);
        return ParseSession(json.RootElement);
    }

    internal async Task<HostedAgentSession> GetAsync(string sessionId, CancellationToken ct)
    {
        using var response = await SendAsync(HttpMethod.Get, SessionUrl(sessionId), null, ct).ConfigureAwait(false);
        using var json = await ParseAsync(response, ct).ConfigureAwait(false);
        return ParseSession(json.RootElement);
    }

    internal async Task<IReadOnlyList<HostedAgentItem>> ListItemsAsync(string sessionId, CancellationToken ct)
    {
        var items = new List<HostedAgentItem>();
        string? after = null;
        do
        {
            var url = SessionUrl(sessionId) + "/items?order=asc&limit=100" +
                (after is null ? "" : "&after=" + Uri.EscapeDataString(after));
            using var response = await SendAsync(HttpMethod.Get, url, null, ct).ConfigureAwait(false);
            using var json = await ParseAsync(response, ct).ConfigureAwait(false);
            var root = json.RootElement;
            if (root.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
            {
                foreach (var element in data.EnumerateArray())
                {
                    items.Add(new HostedAgentItem
                    {
                        Id = String(element, "id") ?? "",
                        Type = String(element, "type") ?? "unknown",
                        Text = ExtractText(element),
                        RawJson = element.GetRawText()
                    });
                }
            }
            var hasMore = root.TryGetProperty("has_more", out var more) && more.ValueKind == JsonValueKind.True;
            var next = String(root, "last_id");
            if (!hasMore) break;
            if (string.IsNullOrEmpty(next) || next == after)
                throw new InvalidDataException("Agent item pagination did not advance.");
            after = next;
        } while (true);
        return items;
    }

    internal async Task<IReadOnlyList<HostedAgentTurn>> ListTurnsAsync(string sessionId, CancellationToken ct)
    {
        var turns = new List<HostedAgentTurn>();
        string? after = null;
        do
        {
            var url = SessionUrl(sessionId) + "/turns?order=asc&limit=100" +
                (after is null ? "" : "&after=" + Uri.EscapeDataString(after));
            using var response = await SendAsync(HttpMethod.Get, url, null, ct).ConfigureAwait(false);
            using var json = await ParseAsync(response, ct).ConfigureAwait(false);
            var root = json.RootElement;
            if (root.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
            {
                foreach (var element in data.EnumerateArray())
                {
                    TokenUsage? usage = null;
                    if (element.TryGetProperty("usage", out var reported) && reported.ValueKind == JsonValueKind.Object)
                    {
                        var inputDetails = reported.TryGetProperty("input_tokens_details", out var input) ? input : default;
                        var outputDetails = reported.TryGetProperty("output_tokens_details", out var output) ? output : default;
                        usage = new TokenUsage
                        {
                            InputTokens = Int(reported, "input_tokens"),
                            OutputTokens = Int(reported, "output_tokens"),
                            CacheReadTokens = Int(inputDetails, "cached_tokens"),
                            ThinkingTokens = Int(outputDetails, "reasoning_tokens")
                        };
                    }
                    turns.Add(new HostedAgentTurn
                    {
                        Id = String(element, "id") ?? "",
                        Status = String(element, "status") ?? "unknown",
                        SubagentId = String(element, "subagent_id"),
                        Usage = usage
                    });
                }
            }
            var hasMore = root.TryGetProperty("has_more", out var more) && more.ValueKind == JsonValueKind.True;
            var next = String(root, "last_id");
            if (!hasMore) break;
            if (string.IsNullOrEmpty(next) || next == after)
                throw new InvalidDataException("Agent turn pagination did not advance.");
            after = next;
        } while (true);
        return turns;
    }

    internal async IAsyncEnumerable<HostedAgentEvent> RunTurnAsync(
        string sessionId, string message, [EnumeratorCancellation] CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(message)) throw new ArgumentException("A message is required.", nameof(message));
        // Stream first: the Agents API does not replay missed events.
        using var request = CreateRequest(HttpMethod.Get, SessionUrl(sessionId) + "/events", null);
        request.Headers.Accept.ParseAdd("text/event-stream");
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, ct).ConfigureAwait(false);
        using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        var input = new JsonObject
        {
            ["events"] = new JsonArray
            {
                (JsonNode)new JsonObject
                {
                    ["type"] = "agent.session.input.message",
                    ["input"] = new JsonArray
                    {
                        (JsonNode)new JsonObject
                        {
                            ["role"] = "user",
                            ["content"] = new JsonArray
                            {
                                (JsonNode)new JsonObject { ["type"] = "input_text", ["text"] = message }
                            }
                        }
                    }
                }
            }
        };
        using (var submitted = await SendAsync(HttpMethod.Post, SessionUrl(sessionId) + "/events", input, ct).ConfigureAwait(false))
            await EnsureSuccessAsync(submitted, ct).ConfigureAwait(false);

        var data = new StringBuilder();
        while (await reader.ReadLineAsync(ct).ConfigureAwait(false) is { } line)
        {
            if (line.Length == 0)
            {
                if (data.Length == 0) continue;
                var jsonText = data.ToString();
                data.Clear();
                using var json = JsonDocument.Parse(jsonText);
                var item = ParseEvent(json.RootElement, jsonText);
                yield return item;
                if (item.Type is "error" or "agent.session.failed" or "agent.session.environment.failed")
                    throw new InvalidOperationException(item.Error ?? item.Type);
                if (item.Type == "agent.session.requires_action")
                    throw new InvalidOperationException("Hosted agent requires an application action. Only hosted tools and service MCP are supported by this mode.");
                if (item.IsRootTurn == true && item.Type is ("agent.session.turn.failed" or "agent.session.turn.cancelled"))
                    throw new InvalidOperationException(item.Error ?? item.Type);
                if (item.IsRootTurn == true && item.Type == "agent.session.turn.completed")
                    yield break;
                continue;
            }
            if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                if (data.Length > 0) data.Append('\n');
                data.Append(line.AsSpan(5).TrimStart());
            }
        }
        throw new IOException("Agent event stream closed before the root turn completed. Retrieve session items to recover; do not resend the task.");
    }

    internal async Task CancelAsync(string sessionId, CancellationToken ct)
    {
        var payload = new JsonObject { ["events"] = new JsonArray { (JsonNode)new JsonObject { ["type"] = "agent.session.input.cancel" } } };
        using var response = await SendAsync(HttpMethod.Post, SessionUrl(sessionId) + "/events", payload, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, ct).ConfigureAwait(false);
    }

    internal async Task DeleteAsync(string sessionId, CancellationToken ct)
    {
        using var response = await SendAsync(HttpMethod.Delete, SessionUrl(sessionId), null, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, ct).ConfigureAwait(false);
    }

    private async Task<HttpResponseMessage> SendAsync(HttpMethod method, string url, JsonObject? payload, CancellationToken ct)
    {
        using var request = CreateRequest(method, url, payload);
        var response = await _httpClient.SendAsync(request, ct).ConfigureAwait(false);
        try { await EnsureSuccessAsync(response, ct).ConfigureAwait(false); return response; }
        catch { response.Dispose(); throw; }
    }

    private HttpRequestMessage CreateRequest(HttpMethod method, string url, JsonObject? payload)
    {
        var request = new HttpRequestMessage(method, url);
        request.Headers.TryAddWithoutValidation("Authorization", "Bearer " + _apiKey);
        request.Headers.TryAddWithoutValidation("OpenAI-Beta", "agents=v1");
        if (payload is not null)
            request.Content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
        return request;
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, CancellationToken ct)
    {
        if (response.IsSuccessStatusCode) return;
        var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
        throw new HttpRequestException($"Agents API returned {(int)response.StatusCode}: {body}", null, response.StatusCode);
    }

    private static async Task<JsonDocument> ParseAsync(HttpResponseMessage response, CancellationToken ct)
    {
        using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        return await JsonDocument.ParseAsync(stream, cancellationToken: ct).ConfigureAwait(false);
    }

    private static string SessionUrl(string sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId)) throw new ArgumentException("Session ID is required.", nameof(sessionId));
        return SessionsUrl + "/" + Uri.EscapeDataString(sessionId);
    }

    private static HostedAgentSession ParseSession(JsonElement root) => new()
    {
        Id = String(root, "id") ?? throw new InvalidDataException("Agents API response has no session ID."),
        Status = String(root, "status") ?? "unknown",
        EnvironmentId = root.TryGetProperty("environment", out var env) ? String(env, "id") : null,
        Error = String(root, "error")
    };

    private static HostedAgentEvent ParseEvent(JsonElement root, string raw)
    {
        var turn = root.TryGetProperty("turn", out var value) ? value : default;
        var isTurn = turn.ValueKind == JsonValueKind.Object;
        return new HostedAgentEvent
        {
            Type = String(root, "type") ?? "unknown",
            RawJson = raw,
            SessionId = String(root, "session_id"),
            TurnId = isTurn ? String(turn, "id") : String(root, "turn_id"),
            ItemId = String(root, "item_id"),
            Delta = String(root, "delta"),
            Text = String(root, "text"),
            Error = root.TryGetProperty("error", out var error)
                ? error.ValueKind == JsonValueKind.Object ? String(error, "message") : error.ToString()
                : isTurn && turn.TryGetProperty("error", out var turnError) ? String(turnError, "message") : null,
            IsRootTurn = isTurn ? !turn.TryGetProperty("subagent_id", out var subagent) || subagent.ValueKind == JsonValueKind.Null : null
        };
    }

    private static string? ExtractText(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object) return null;
        if (element.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.Array)
        {
            var text = new StringBuilder();
            foreach (var part in content.EnumerateArray())
            {
                var value = String(part, "text");
                if (!string.IsNullOrEmpty(value)) text.Append(value);
            }
            if (text.Length > 0) return text.ToString();
        }
        return String(element, "text") ?? String(element, "output");
    }

    private static string? String(JsonElement element, string name)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() : null;

    private static int Int(JsonElement element, string name)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) &&
           value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var result) ? result : 0;
}
