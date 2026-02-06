using System.Buffers;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ai.Tlbx.Inference.Providers;

internal abstract class OpenAiCompatibleProvider : IProvider
{
    private readonly ProviderRequestContext _context;

    protected OpenAiCompatibleProvider(ProviderRequestContext context)
    {
        _context = context;
    }

    protected abstract string MapReasoningEffort(int thinkingBudget);

    public async Task<ProviderResponse> CompleteAsync(ProviderRequest request, CancellationToken ct)
    {
        var body = BuildRequestBody(request, stream: false);
        var jsonBytes = SerializeToUtf8Bytes(body);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {_context.BaseUrl}/v1/chat/completions");

        using var httpRequest = CreateHttpRequest(jsonBytes, "/v1/chat/completions");
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"API request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;

        var choice = root.GetProperty("choices")[0];
        var message = choice.GetProperty("message");

        var content = message.TryGetProperty("content", out var contentEl) && contentEl.ValueKind != JsonValueKind.Null
            ? contentEl.GetString() ?? ""
            : "";

        var stopReason = choice.TryGetProperty("finish_reason", out var finishEl)
            ? finishEl.GetString()
            : null;

        List<ToolCallRequest>? toolCalls = null;
        if (message.TryGetProperty("tool_calls", out var toolCallsEl))
        {
            toolCalls = [];
            foreach (var tc in toolCallsEl.EnumerateArray())
            {
                var function = tc.GetProperty("function");
                toolCalls.Add(new ToolCallRequest
                {
                    Id = tc.GetProperty("id").GetString()!,
                    Name = function.GetProperty("name").GetString()!,
                    Arguments = function.GetProperty("arguments").GetString()!,
                });
            }
        }

        var usage = ParseUsage(root.GetProperty("usage"));

        return new ProviderResponse
        {
            Content = content,
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

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {_context.BaseUrl}/v1/chat/completions");

        using var httpRequest = CreateHttpRequest(jsonBytes, "/v1/chat/completions");
        using var response = await _context.HttpClient.SendAsync(
            httpRequest,
            HttpCompletionOption.ResponseHeadersRead,
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"API stream request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);

        var toolCallAccumulator = new Dictionary<int, (string Id, string Name, StringBuilder Arguments)>();

        await foreach (var data in SseStreamParser.ParseAsync(stream, ct).ConfigureAwait(false))
        {
            using var doc = JsonDocument.Parse(data);
            var root = doc.RootElement;

            if (root.TryGetProperty("usage", out var usageEl))
            {
                yield return new ProviderStreamEvent
                {
                    Usage = ParseUsage(usageEl),
                };
            }

            if (!root.TryGetProperty("choices", out var choices) || choices.GetArrayLength() == 0)
            {
                continue;
            }

            var choice = choices[0];

            if (!choice.TryGetProperty("delta", out var delta))
            {
                continue;
            }

            if (delta.TryGetProperty("content", out var contentEl) && contentEl.ValueKind == JsonValueKind.String)
            {
                var text = contentEl.GetString();
                if (text is not null)
                {
                    yield return new ProviderStreamEvent { TextDelta = text };
                }
            }

            if (delta.TryGetProperty("tool_calls", out var toolCallsDelta))
            {
                foreach (var tc in toolCallsDelta.EnumerateArray())
                {
                    var index = tc.GetProperty("index").GetInt32();

                    if (tc.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String)
                    {
                        var function = tc.GetProperty("function");
                        toolCallAccumulator[index] = (
                            idEl.GetString()!,
                            function.GetProperty("name").GetString()!,
                            new StringBuilder(function.TryGetProperty("arguments", out var argsEl)
                                ? argsEl.GetString() ?? ""
                                : "")
                        );
                    }
                    else if (toolCallAccumulator.TryGetValue(index, out var existing))
                    {
                        if (tc.TryGetProperty("function", out var fnEl) &&
                            fnEl.TryGetProperty("arguments", out var argChunk) &&
                            argChunk.ValueKind == JsonValueKind.String)
                        {
                            existing.Arguments.Append(argChunk.GetString());
                        }
                    }
                }
            }

            if (choice.TryGetProperty("finish_reason", out var finishEl) &&
                finishEl.ValueKind == JsonValueKind.String &&
                finishEl.GetString() == "tool_calls")
            {
                foreach (var (_, (id, name, args)) in toolCallAccumulator)
                {
                    yield return new ProviderStreamEvent
                    {
                        ToolCall = new ToolCallRequest
                        {
                            Id = id,
                            Name = name,
                            Arguments = args.ToString(),
                        },
                    };
                }

                toolCallAccumulator.Clear();
            }
        }
    }

    public virtual async Task<ProviderEmbeddingResponse> EmbedAsync(
        ProviderEmbeddingRequest request,
        CancellationToken ct)
    {
        var body = BuildEmbeddingBody(request.ModelApiName, request.Input, request.Dimensions);
        var jsonBytes = SerializeToUtf8Bytes(body);

        using var httpRequest = CreateHttpRequest(jsonBytes, "/v1/embeddings");
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Embedding request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;

        var embedding = ParseEmbeddingArray(root.GetProperty("data")[0].GetProperty("embedding"));
        var promptTokens = root.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();

        return new ProviderEmbeddingResponse
        {
            Embedding = embedding,
            Usage = new TokenUsage { InputTokens = promptTokens },
        };
    }

    public virtual async Task<ProviderBatchEmbeddingResponse> EmbedBatchAsync(
        ProviderBatchEmbeddingRequest request,
        CancellationToken ct)
    {
        var body = BuildBatchEmbeddingBody(request.ModelApiName, request.Inputs, request.Dimensions);
        var jsonBytes = SerializeToUtf8Bytes(body);

        using var httpRequest = CreateHttpRequest(jsonBytes, "/v1/embeddings");
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Batch embedding request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;

        var dataArray = root.GetProperty("data");
        var embeddings = new List<ReadOnlyMemory<float>>(dataArray.GetArrayLength());

        foreach (var item in dataArray.EnumerateArray())
        {
            embeddings.Add(ParseEmbeddingArray(item.GetProperty("embedding")));
        }

        var promptTokens = root.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();

        return new ProviderBatchEmbeddingResponse
        {
            Embeddings = embeddings,
            Usage = new TokenUsage { InputTokens = promptTokens },
        };
    }

    public virtual Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
        => throw new NotSupportedException("Image generation is not supported by this provider.");

    private JsonObject BuildRequestBody(ProviderRequest request, bool stream)
    {
        var messages = new JsonArray();

        if (request.SystemMessage is not null)
        {
            messages.Add((JsonNode)new JsonObject { ["role"] = "system", ["content"] = request.SystemMessage });
        }

        foreach (var msg in request.Messages)
        {
            switch (msg.Role)
            {
                case ChatRole.System:
                    messages.Add((JsonNode)new JsonObject { ["role"] = "system", ["content"] = msg.Content ?? "" });
                    break;

                case ChatRole.User:
                    messages.Add((JsonNode)new JsonObject { ["role"] = "user", ["content"] = msg.Content ?? "" });
                    break;

                case ChatRole.Assistant when msg.ToolCalls is { Count: > 0 }:
                    var toolCallsArray = new JsonArray();
                    foreach (var tc in msg.ToolCalls)
                    {
                        toolCallsArray.Add((JsonNode)new JsonObject
                        {
                            ["id"] = tc.Id,
                            ["type"] = "function",
                            ["function"] = new JsonObject { ["name"] = tc.Name, ["arguments"] = tc.Arguments },
                        });
                    }
                    messages.Add((JsonNode)new JsonObject { ["role"] = "assistant", ["content"] = msg.Content, ["tool_calls"] = toolCallsArray });
                    break;

                case ChatRole.Assistant:
                    messages.Add((JsonNode)new JsonObject { ["role"] = "assistant", ["content"] = msg.Content ?? "" });
                    break;

                case ChatRole.Tool:
                    messages.Add((JsonNode)new JsonObject { ["role"] = "tool", ["content"] = msg.Content ?? "", ["tool_call_id"] = msg.ToolCallId });
                    break;
            }
        }

        var body = new JsonObject
        {
            ["model"] = request.ModelApiName,
            ["messages"] = messages,
        };

        if (request.Temperature.HasValue)
        {
            body["temperature"] = request.Temperature.Value;
        }

        if (request.MaxTokens.HasValue)
        {
            body["max_completion_tokens"] = request.MaxTokens.Value;
        }

        if (request.TopP.HasValue)
        {
            body["top_p"] = request.TopP.Value;
        }

        if (request.StopSequences is { Count: > 0 })
        {
            var stopArray = new JsonArray();
            foreach (var s in request.StopSequences)
            {
                stopArray.Add((JsonNode)JsonValue.Create(s)!);
            }
            body["stop"] = stopArray;
        }

        if (request.ThinkingBudget.HasValue)
        {
            body["reasoning_effort"] = MapReasoningEffort(request.ThinkingBudget.Value);
        }

        if (request.Tools is { Count: > 0 })
        {
            var toolsArray = new JsonArray();
            foreach (var t in request.Tools)
            {
                toolsArray.Add((JsonNode)new JsonObject
                {
                    ["type"] = "function",
                    ["function"] = new JsonObject
                    {
                        ["name"] = t.Name,
                        ["description"] = t.Description,
                        ["parameters"] = JsonNode.Parse(t.ParametersSchema.GetRawText()),
                    },
                });
            }
            body["tools"] = toolsArray;
        }

        if (request.JsonSchema is not null)
        {
            body["response_format"] = new JsonObject
            {
                ["type"] = "json_schema",
                ["json_schema"] = new JsonObject
                {
                    ["name"] = "response",
                    ["strict"] = true,
                    ["schema"] = JsonNode.Parse(request.JsonSchema),
                },
            };
        }

        if (stream)
        {
            body["stream"] = true;
            body["stream_options"] = new JsonObject { ["include_usage"] = true };
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

    private HttpRequestMessage CreateHttpRequest(ReadOnlyMemory<byte> jsonBytes, string path)
    {
        var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_context.BaseUrl}{path}")
        {
            Content = new ReadOnlyMemoryContent(jsonBytes),
        };
        httpRequest.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        httpRequest.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_context.ApiKey}");
        return httpRequest;
    }

    private static TokenUsage ParseUsage(JsonElement usage)
    {
        var inputTokens = usage.TryGetProperty("prompt_tokens", out var pt) ? pt.GetInt32() : 0;
        var outputTokens = usage.TryGetProperty("completion_tokens", out var ct) ? ct.GetInt32() : 0;

        var cacheReadTokens = 0;
        if (usage.TryGetProperty("prompt_tokens_details", out var ptd) &&
            ptd.TryGetProperty("cached_tokens", out var cached))
        {
            cacheReadTokens = cached.GetInt32();
        }

        var thinkingTokens = 0;
        if (usage.TryGetProperty("completion_tokens_details", out var ctd) &&
            ctd.TryGetProperty("reasoning_tokens", out var reasoning))
        {
            thinkingTokens = reasoning.GetInt32();
        }

        return new TokenUsage
        {
            InputTokens = inputTokens,
            OutputTokens = outputTokens,
            CacheReadTokens = cacheReadTokens,
            ThinkingTokens = thinkingTokens,
        };
    }

    private static float[] ParseEmbeddingArray(JsonElement embeddingEl)
    {
        var arr = new float[embeddingEl.GetArrayLength()];
        var i = 0;
        foreach (var val in embeddingEl.EnumerateArray())
        {
            arr[i++] = val.GetSingle();
        }
        return arr;
    }

    private static JsonObject BuildEmbeddingBody(string model, string input, int? dimensions)
    {
        var body = new JsonObject
        {
            ["model"] = model,
            ["input"] = input,
        };

        if (dimensions.HasValue)
        {
            body["dimensions"] = dimensions.Value;
        }

        return body;
    }

    private static JsonObject BuildBatchEmbeddingBody(string model, IReadOnlyList<string> inputs, int? dimensions)
    {
        var inputArray = new JsonArray();
        foreach (var input in inputs)
        {
            inputArray.Add((JsonNode)JsonValue.Create(input)!);
        }

        var body = new JsonObject
        {
            ["model"] = model,
            ["input"] = inputArray,
        };

        if (dimensions.HasValue)
        {
            body["dimensions"] = dimensions.Value;
        }

        return body;
    }
}
