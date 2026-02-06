using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace Ai.Tlbx.Inference.Providers;

internal abstract class OpenAiCompatibleProvider : IProvider
{
    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    private readonly ProviderRequestContext _context;

    protected OpenAiCompatibleProvider(ProviderRequestContext context)
    {
        _context = context;
    }

    protected abstract string MapReasoningEffort(int thinkingBudget);

    public async Task<ProviderResponse> CompleteAsync(ProviderRequest request, CancellationToken ct)
    {
        var body = BuildRequestBody(request, stream: false);
        var json = JsonSerializer.Serialize(body, _jsonOptions);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {_context.BaseUrl}/v1/chat/completions");

        using var httpRequest = CreateHttpRequest(json, "/v1/chat/completions");
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

        var responseBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"API request failed with status {response.StatusCode}: {responseBody}",
                null,
                response.StatusCode);
        }

        using var doc = JsonDocument.Parse(responseBody);
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
        var json = JsonSerializer.Serialize(body, _jsonOptions);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {_context.BaseUrl}/v1/chat/completions");

        using var httpRequest = CreateHttpRequest(json, "/v1/chat/completions");
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
        var json = JsonSerializer.Serialize(body, _jsonOptions);

        using var httpRequest = CreateHttpRequest(json, "/v1/embeddings");
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

        var responseBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Embedding request failed with status {response.StatusCode}: {responseBody}",
                null,
                response.StatusCode);
        }

        using var doc = JsonDocument.Parse(responseBody);
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
        var json = JsonSerializer.Serialize(body, _jsonOptions);

        using var httpRequest = CreateHttpRequest(json, "/v1/embeddings");
        using var response = await _context.HttpClient.SendAsync(httpRequest, ct).ConfigureAwait(false);

        var responseBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Batch embedding request failed with status {response.StatusCode}: {responseBody}",
                null,
                response.StatusCode);
        }

        using var doc = JsonDocument.Parse(responseBody);
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

    private Dictionary<string, object?> BuildRequestBody(ProviderRequest request, bool stream)
    {
        var messages = new List<object>();

        if (request.SystemMessage is not null)
        {
            messages.Add(new { role = "system", content = request.SystemMessage });
        }

        foreach (var msg in request.Messages)
        {
            switch (msg.Role)
            {
                case ChatRole.System:
                    messages.Add(new { role = "system", content = msg.Content ?? "" });
                    break;

                case ChatRole.User:
                    messages.Add(new { role = "user", content = msg.Content ?? "" });
                    break;

                case ChatRole.Assistant when msg.ToolCalls is { Count: > 0 }:
                    var toolCalls = msg.ToolCalls.Select(tc => new
                    {
                        id = tc.Id,
                        type = "function",
                        function = new { name = tc.Name, arguments = tc.Arguments },
                    }).ToList();
                    messages.Add(new { role = "assistant", content = msg.Content, tool_calls = toolCalls });
                    break;

                case ChatRole.Assistant:
                    messages.Add(new { role = "assistant", content = msg.Content ?? "" });
                    break;

                case ChatRole.Tool:
                    messages.Add(new { role = "tool", content = msg.Content ?? "", tool_call_id = msg.ToolCallId });
                    break;
            }
        }

        var body = new Dictionary<string, object?>
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
            body["stop"] = request.StopSequences;
        }

        if (request.ThinkingBudget.HasValue)
        {
            body["reasoning_effort"] = MapReasoningEffort(request.ThinkingBudget.Value);
        }

        if (request.Tools is { Count: > 0 })
        {
            body["tools"] = request.Tools.Select(t => new
            {
                type = "function",
                function = new
                {
                    name = t.Name,
                    description = t.Description,
                    parameters = t.ParametersSchema,
                },
            }).ToList();
        }

        if (request.JsonSchema is not null)
        {
            using var schemaDoc = JsonDocument.Parse(request.JsonSchema);
            body["response_format"] = new
            {
                type = "json_schema",
                json_schema = new
                {
                    name = "response",
                    strict = true,
                    schema = schemaDoc.RootElement.Clone(),
                },
            };
        }

        if (stream)
        {
            body["stream"] = true;
            body["stream_options"] = new { include_usage = true };
        }

        return body;
    }

    private HttpRequestMessage CreateHttpRequest(string json, string path)
    {
        var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_context.BaseUrl}{path}")
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };

        httpRequest.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _context.ApiKey);
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

    private static object BuildEmbeddingBody(string model, string input, int? dimensions)
    {
        if (dimensions.HasValue)
        {
            return new { model, input, dimensions = dimensions.Value };
        }
        return new { model, input };
    }

    private static object BuildBatchEmbeddingBody(string model, IReadOnlyList<string> inputs, int? dimensions)
    {
        if (dimensions.HasValue)
        {
            return new { model, input = inputs, dimensions = dimensions.Value };
        }
        return new { model, input = inputs };
    }
}
