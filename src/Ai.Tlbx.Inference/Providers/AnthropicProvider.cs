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

        using var httpRequest = CreateHttpRequest(jsonBytes);
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

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

        return new ProviderResponse
        {
            Content = contentBuilder.ToString(),
            Usage = usage,
            StopReason = stopReason,
            ToolCalls = toolCalls,
        };
    }

    public async IAsyncEnumerable<ProviderStreamEvent> StreamAsync(
        ProviderRequest request,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var body = BuildRequestBody(request, stream: true);
        var jsonBytes = SerializeToUtf8Bytes(body);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {_context.BaseUrl}/v1/messages");

        using var httpRequest = CreateHttpRequest(jsonBytes);
        using var response = await _context.HttpClient.SendAsync(
            httpRequest,
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

        while (!reader.EndOfStream)
        {
            ct.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);

            if (line is null)
            {
                break;
            }

            if (line.StartsWith("event: ", StringComparison.Ordinal))
            {
                var eventType = line["event: ".Length..];

                if (eventType == "message_stop")
                {
                    yield return new ProviderStreamEvent { Usage = accumulatedUsage };
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

        if (request.Tools is { Count: > 0 })
        {
            var toolsArray = new JsonArray();
            foreach (var t in request.Tools)
            {
                toolsArray.Add((JsonNode)new JsonObject
                {
                    ["name"] = t.Name,
                    ["description"] = t.Description,
                    ["input_schema"] = JsonNode.Parse(t.ParametersSchema.GetRawText()),
                });
            }
            body["tools"] = toolsArray;
        }

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
}
