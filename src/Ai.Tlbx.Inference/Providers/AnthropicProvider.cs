using System.Buffers;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class AnthropicProvider : IProvider
{
    private readonly ProviderRequestContext _context;

    public AnthropicProvider(ProviderRequestContext context)
    {
        _context = context;
    }

    public async Task<ProviderResponse> CompleteAsync(ProviderRequest request, CancellationToken ct)
    {
        var body = BuildRequestBody(request, stream: false);
        var jsonBytes = SerializeToUtf8Bytes(body);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {_context.BaseUrl}/v1/messages");

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Anthropic request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;

        var contentBuilder = new StringBuilder();
        List<ToolCallRequest>? toolCalls = null;

        if (root.TryGetProperty("content", out var contentArray))
        {
            foreach (var block in contentArray.EnumerateArray())
            {
                var type = block.GetProperty("type").GetString();

                if (type == "text")
                {
                    contentBuilder.Append(block.GetProperty("text").GetString());
                }
                else if (type == "tool_use")
                {
                    toolCalls ??= [];
                    toolCalls.Add(new ToolCallRequest
                    {
                        Id = block.GetProperty("id").GetString()!,
                        Name = block.GetProperty("name").GetString()!,
                        Arguments = block.GetProperty("input").GetRawText(),
                    });
                }
            }
        }

        var stopReason = root.TryGetProperty("stop_reason", out var stopEl)
            ? stopEl.GetString()
            : null;

        var usage = ParseUsage(root.GetProperty("usage"));
        var grounding = ParseGrounding(root);

        return new ProviderResponse
        {
            Content = contentBuilder.ToString(),
            Usage = usage,
            EndpointFamily = ModelEndpointFamily.AnthropicMessages,
            StopReason = stopReason,
            ToolCalls = toolCalls,
            DiagnosticNote = BuildDiagnosticNote(contentBuilder.ToString(), stopReason),
            Grounding = grounding,
        };
    }

    public async IAsyncEnumerable<ProviderStreamEvent> StreamAsync(
        ProviderRequest request,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var body = BuildRequestBody(request, stream: true);
        var jsonBytes = SerializeToUtf8Bytes(body);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {_context.BaseUrl}/v1/messages");

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes),
            HttpCompletionOption.ResponseHeadersRead,
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Anthropic stream request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        string? currentBlockType = null;
        string? currentToolId = null;
        string? currentToolName = null;
        var toolArgsBuilder = new StringBuilder();
        var accumulatedUsage = new TokenUsage();
        var groundingSources = new List<GroundingSource>();
        var groundingQueries = new List<string>();
        var webSearchCalls = 0;

        string? line;
        while ((line = await reader.ReadLineAsync(ct).ConfigureAwait(false)) is not null)
        {
            ct.ThrowIfCancellationRequested();

            if (line.StartsWith("event: ", StringComparison.Ordinal))
            {
                var eventType = line["event: ".Length..];

                if (eventType == "message_stop")
                {
                    var grounding = groundingSources.Count == 0 && groundingQueries.Count == 0 && webSearchCalls == 0
                        ? null
                        : new GroundingResult
                        {
                            Sources = groundingSources.DistinctBy(source => source.Url, StringComparer.OrdinalIgnoreCase).ToArray(),
                            SearchQueries = groundingQueries.Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
                            Usage = new GroundingUsage { WebSearchCalls = webSearchCalls },
                        };
                    yield return new ProviderStreamEvent { Usage = accumulatedUsage, Grounding = grounding };
                    yield break;
                }

                continue;
            }

            if (!line.StartsWith("data: ", StringComparison.Ordinal))
            {
                continue;
            }

            var data = line["data: ".Length..];

            using var doc = JsonDocument.Parse(data);
            var root = doc.RootElement;
            var type = root.GetProperty("type").GetString();

            switch (type)
            {
                case "message_start":
                {
                    if (root.TryGetProperty("message", out var message) &&
                        message.TryGetProperty("usage", out var usageEl))
                    {
                        var inputTokens = usageEl.TryGetProperty("input_tokens", out var it)
                            ? it.GetInt32() : 0;
                        var cacheRead = usageEl.TryGetProperty("cache_read_input_tokens", out var cr)
                            ? cr.GetInt32() : 0;
                        var cacheWrite = usageEl.TryGetProperty("cache_creation_input_tokens", out var cw)
                            ? cw.GetInt32() : 0;

                        accumulatedUsage = accumulatedUsage with
                        {
                            InputTokens = inputTokens,
                            CacheReadTokens = cacheRead,
                            CacheWriteTokens = cacheWrite,
                        };
                        webSearchCalls = ParseWebSearchRequestCount(usageEl);
                    }
                    break;
                }

                case "content_block_start":
                {
                    if (root.TryGetProperty("content_block", out var block))
                    {
                        currentBlockType = block.GetProperty("type").GetString();

                        if (currentBlockType == "tool_use")
                        {
                            currentToolId = block.GetProperty("id").GetString();
                            currentToolName = block.GetProperty("name").GetString();
                            toolArgsBuilder.Clear();
                        }
                        else if (currentBlockType == "web_search_tool_result")
                        {
                            ParseAnthropicSearchResults(block, groundingSources);
                        }
                        else if (currentBlockType == "server_tool_use" &&
                                 block.TryGetProperty("name", out var serverName) && serverName.GetString() == "web_search" &&
                                 block.TryGetProperty("input", out var input) && input.TryGetProperty("query", out var query) &&
                                 !string.IsNullOrWhiteSpace(query.GetString()))
                        {
                            groundingQueries.Add(query.GetString()!);
                        }
                    }
                    break;
                }

                case "content_block_delta":
                {
                    if (root.TryGetProperty("delta", out var delta))
                    {
                        var deltaType = delta.GetProperty("type").GetString();

                        if (deltaType == "text_delta")
                        {
                            var text = delta.GetProperty("text").GetString();
                            if (text is not null)
                            {
                                yield return new ProviderStreamEvent { TextDelta = text };
                            }
                        }
                        else if (deltaType == "input_json_delta")
                        {
                            var partial = delta.GetProperty("partial_json").GetString();
                            if (partial is not null)
                            {
                                toolArgsBuilder.Append(partial);
                            }
                        }
                    }
                    break;
                }

                case "content_block_stop":
                {
                    if (currentBlockType == "tool_use" && currentToolId is not null && currentToolName is not null)
                    {
                        yield return new ProviderStreamEvent
                        {
                            ToolCall = new ToolCallRequest
                            {
                                Id = currentToolId,
                                Name = currentToolName,
                                Arguments = toolArgsBuilder.ToString(),
                            },
                        };
                    }

                    currentBlockType = null;
                    currentToolId = null;
                    currentToolName = null;
                    toolArgsBuilder.Clear();
                    break;
                }

                case "message_delta":
                {
                    if (root.TryGetProperty("usage", out var usageEl))
                    {
                        var outputTokens = usageEl.TryGetProperty("output_tokens", out var ot)
                            ? ot.GetInt32() : 0;

                        accumulatedUsage = accumulatedUsage with
                        {
                            OutputTokens = outputTokens,
                        };
                        webSearchCalls = Math.Max(webSearchCalls, ParseWebSearchRequestCount(usageEl));
                    }
                    break;
                }
            }
        }
    }

    public Task<ProviderEmbeddingResponse> EmbedAsync(ProviderEmbeddingRequest request, CancellationToken ct)
        => throw new NotSupportedException("Anthropic does not support embeddings.");

    public Task<ProviderBatchEmbeddingResponse> EmbedBatchAsync(ProviderBatchEmbeddingRequest request, CancellationToken ct)
        => throw new NotSupportedException("Anthropic does not support embeddings.");

    public Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
        => throw new NotSupportedException("Anthropic does not support image generation.");

    public Task<byte[]> EditImageAsync(ProviderImageEditRequest request, CancellationToken ct)
        => throw new NotSupportedException("Anthropic does not support image editing.");

    private JsonObject BuildRequestBody(ProviderRequest request, bool stream)
    {
        var messages = new JsonArray();

        foreach (var msg in request.Messages)
        {
            switch (msg.Role)
            {
                case ChatRole.User when msg.Attachments is { Count: > 0 }:
                {
                    var parts = new JsonArray();

                    foreach (var attachment in msg.Attachments)
                    {
                        var base64 = Convert.ToBase64String(attachment.Content.Span);
                        var contentType = attachment.MimeType.StartsWith("image/", StringComparison.OrdinalIgnoreCase)
                            ? "image"
                            : "document";

                        parts.Add((JsonNode)new JsonObject
                        {
                            ["type"] = contentType,
                            ["source"] = new JsonObject
                            {
                                ["type"] = "base64",
                                ["media_type"] = attachment.MimeType,
                                ["data"] = base64,
                            },
                        });
                    }

                    if (!string.IsNullOrEmpty(msg.Content))
                    {
                        parts.Add((JsonNode)new JsonObject { ["type"] = "text", ["text"] = msg.Content });
                    }

                    messages.Add((JsonNode)new JsonObject { ["role"] = "user", ["content"] = parts });
                    break;
                }

                case ChatRole.User:
                    messages.Add((JsonNode)new JsonObject { ["role"] = "user", ["content"] = msg.Content ?? "" });
                    break;

                case ChatRole.Assistant when msg.ToolCalls is { Count: > 0 }:
                {
                    var parts = new JsonArray();
                    if (!string.IsNullOrEmpty(msg.Content))
                    {
                        parts.Add((JsonNode)new JsonObject { ["type"] = "text", ["text"] = msg.Content });
                    }
                    foreach (var tc in msg.ToolCalls)
                    {
                        parts.Add((JsonNode)new JsonObject
                        {
                            ["type"] = "tool_use",
                            ["id"] = tc.Id,
                            ["name"] = tc.Name,
                            ["input"] = JsonNode.Parse(tc.Arguments),
                        });
                    }
                    messages.Add((JsonNode)new JsonObject { ["role"] = "assistant", ["content"] = parts });
                    break;
                }

                case ChatRole.Assistant:
                    messages.Add((JsonNode)new JsonObject { ["role"] = "assistant", ["content"] = msg.Content ?? "" });
                    break;

                case ChatRole.Tool:
                    messages.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = new JsonArray
                        {
                            (JsonNode)new JsonObject
                            {
                                ["type"] = "tool_result",
                                ["tool_use_id"] = msg.ToolCallId,
                                ["content"] = msg.Content ?? "",
                            },
                        },
                    });
                    break;
            }
        }

        var body = new JsonObject
        {
            ["model"] = request.ModelApiName,
            ["max_tokens"] = request.MaxTokens ?? 8192,
            ["messages"] = messages,
        };

        if (request.SystemMessage is not null)
        {
            if (request.EnableCache)
            {
                body["system"] = new JsonArray
                {
                    (JsonNode)new JsonObject
                    {
                        ["type"] = "text",
                        ["text"] = request.SystemMessage,
                        ["cache_control"] = new JsonObject { ["type"] = "ephemeral" },
                    },
                };
            }
            else
            {
                body["system"] = request.SystemMessage;
            }
        }

        if (request.Temperature.HasValue)
        {
            body["temperature"] = request.Temperature.Value;
        }

        if (request.TopP.HasValue)
        {
            body["top_p"] = request.TopP.Value;
        }

        if (request.StopSequences is { Count: > 0 })
        {
            var stopArray = new JsonArray();
            foreach (var seq in request.StopSequences)
            {
                stopArray.Add((JsonNode)JsonValue.Create(seq)!);
            }
            body["stop_sequences"] = stopArray;
        }

        if (request.ThinkingBudget.HasValue)
        {
            body["thinking"] = new JsonObject
            {
                ["type"] = "enabled",
                ["budget_tokens"] = request.ThinkingBudget.Value,
            };

            var maxTokens = request.MaxTokens ?? 8192;
            if (maxTokens <= request.ThinkingBudget.Value)
            {
                body["max_tokens"] = request.ThinkingBudget.Value + 4096;
            }
        }

        var toolsArray = new JsonArray();
        if (request.EnableWebSearch || request.Grounding?.EnableWebSearch == true)
        {
            var grounding = request.Grounding;
            var webSearch = new JsonObject
            {
                ["type"] = "web_search_20250305",
                ["name"] = "web_search",
                ["max_uses"] = Math.Clamp(grounding?.MaxSearches ?? 3, 1, 20),
            };
            if (grounding?.AllowedDomains is { Count: > 0 })
            {
                webSearch["allowed_domains"] = ToJsonArray(grounding.AllowedDomains);
            }
            else if (grounding?.BlockedDomains is { Count: > 0 })
            {
                webSearch["blocked_domains"] = ToJsonArray(grounding.BlockedDomains);
            }
            if (grounding?.UserLocation is { } location)
            {
                var userLocation = new JsonObject { ["type"] = "approximate" };
                if (!string.IsNullOrWhiteSpace(location.City)) userLocation["city"] = location.City;
                if (!string.IsNullOrWhiteSpace(location.Region)) userLocation["region"] = location.Region;
                if (!string.IsNullOrWhiteSpace(location.Country)) userLocation["country"] = location.Country;
                if (!string.IsNullOrWhiteSpace(location.Timezone)) userLocation["timezone"] = location.Timezone;
                webSearch["user_location"] = userLocation;
            }
            toolsArray.Add((JsonNode)webSearch);
        }

        if (request.Tools is { Count: > 0 })
        {
            foreach (var t in request.Tools)
            {
                toolsArray.Add((JsonNode)new JsonObject
                {
                    ["name"] = t.Name,
                    ["description"] = t.Description,
                    ["input_schema"] = JsonNode.Parse(t.ParametersSchema.GetRawText()),
                });
            }
        }

        if (toolsArray.Count > 0) body["tools"] = toolsArray;

        if (request.JsonSchema is not null)
        {
            body["tools"] = new JsonArray
            {
                (JsonNode)new JsonObject
                {
                    ["name"] = "json_response",
                    ["description"] = "Respond with structured JSON matching the provided schema.",
                    ["input_schema"] = JsonNode.Parse(request.JsonSchema),
                },
            };
            body["tool_choice"] = new JsonObject { ["type"] = "tool", ["name"] = "json_response" };
        }

        if (stream)
        {
            body["stream"] = true;
        }

        return body;
    }

    private static ReadOnlyMemory<byte> SerializeToUtf8Bytes(JsonObject body)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using var writer = new Utf8JsonWriter(buffer);
        body.WriteTo(writer);
        writer.Flush();
        return buffer.WrittenMemory;
    }

    private HttpRequestMessage CreateHttpRequest(ReadOnlyMemory<byte> jsonBytes)
    {
        var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_context.BaseUrl}/v1/messages")
        {
            Content = new ReadOnlyMemoryContent(jsonBytes),
        };
        httpRequest.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        httpRequest.Headers.TryAddWithoutValidation("x-api-key", _context.ApiKey);
        httpRequest.Headers.TryAddWithoutValidation("anthropic-version", "2023-06-01");

        return httpRequest;
    }

    private static TokenUsage ParseUsage(JsonElement usage)
    {
        var inputTokens = usage.TryGetProperty("input_tokens", out var it) ? it.GetInt32() : 0;
        var outputTokens = usage.TryGetProperty("output_tokens", out var ot) ? ot.GetInt32() : 0;
        var cacheRead = usage.TryGetProperty("cache_read_input_tokens", out var cr) ? cr.GetInt32() : 0;
        var cacheWrite = usage.TryGetProperty("cache_creation_input_tokens", out var cw) ? cw.GetInt32() : 0;

        return new TokenUsage
        {
            InputTokens = inputTokens,
            OutputTokens = outputTokens,
            CacheReadTokens = cacheRead,
            CacheWriteTokens = cacheWrite,
        };
    }

    private static GroundingResult? ParseGrounding(JsonElement root)
    {
        var sources = new List<GroundingSource>();
        var queries = new List<string>();
        if (root.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.Array)
        {
            foreach (var block in content.EnumerateArray())
            {
                var type = block.TryGetProperty("type", out var typeEl) ? typeEl.GetString() : null;
                if (type == "text" && block.TryGetProperty("citations", out var citations) && citations.ValueKind == JsonValueKind.Array)
                {
                    foreach (var citation in citations.EnumerateArray())
                    {
                        var url = citation.TryGetProperty("url", out var urlEl) ? urlEl.GetString() : null;
                        if (string.IsNullOrWhiteSpace(url)) continue;
                        sources.Add(new GroundingSource
                        {
                            Url = url!,
                            Title = citation.TryGetProperty("title", out var title) ? title.GetString() : null,
                            CitedText = citation.TryGetProperty("cited_text", out var cited) ? cited.GetString() : null,
                        });
                    }
                }
                else if (type == "web_search_tool_result")
                {
                    ParseAnthropicSearchResults(block, sources);
                }
                else if (type == "server_tool_use" && block.TryGetProperty("name", out var name) &&
                         name.GetString() == "web_search" && block.TryGetProperty("input", out var input) &&
                         input.TryGetProperty("query", out var query) && !string.IsNullOrWhiteSpace(query.GetString()))
                {
                    queries.Add(query.GetString()!);
                }
            }
        }

        var calls = root.TryGetProperty("usage", out var usage) ? ParseWebSearchRequestCount(usage) : 0;
        if (sources.Count == 0 && queries.Count == 0 && calls == 0) return null;
        return new GroundingResult
        {
            Sources = sources.DistinctBy(source => source.Url, StringComparer.OrdinalIgnoreCase).ToArray(),
            SearchQueries = queries.Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
            Usage = new GroundingUsage { WebSearchCalls = calls },
        };
    }

    private static void ParseAnthropicSearchResults(JsonElement block, List<GroundingSource> sources)
    {
        if (!block.TryGetProperty("content", out var results) || results.ValueKind != JsonValueKind.Array) return;
        foreach (var result in results.EnumerateArray())
        {
            var url = result.TryGetProperty("url", out var urlEl) ? urlEl.GetString() : null;
            if (string.IsNullOrWhiteSpace(url)) continue;
            sources.Add(new GroundingSource
            {
                Url = url!,
                Title = result.TryGetProperty("title", out var title) ? title.GetString() : null,
            });
        }
    }

    private static int ParseWebSearchRequestCount(JsonElement usage)
        => usage.TryGetProperty("server_tool_use", out var serverTools) &&
           serverTools.TryGetProperty("web_search_requests", out var count)
            ? count.GetInt32()
            : 0;

    private static JsonArray ToJsonArray(IEnumerable<string> values)
    {
        var array = new JsonArray();
        foreach (var value in values.Where(value => !string.IsNullOrWhiteSpace(value)))
        {
            array.Add((JsonNode)JsonValue.Create(value.Trim())!);
        }
        return array;
    }

    private static string? BuildDiagnosticNote(string content, string? stopReason)
    {
        if (string.IsNullOrWhiteSpace(content) && string.Equals(stopReason, "max_tokens", StringComparison.OrdinalIgnoreCase))
        {
            return "Anthropic stopped at the output token limit before returning visible text.";
        }

        return null;
    }
}
