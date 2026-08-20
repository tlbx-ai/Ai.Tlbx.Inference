using System.Buffers;
using System.Net;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ai.Tlbx.Inference.Json;

namespace Ai.Tlbx.Inference.Providers;

internal abstract class OpenAiCompatibleProvider : IProvider
{
    protected readonly ProviderRequestContext _context;

    protected OpenAiCompatibleProvider(ProviderRequestContext context)
    {
        _context = context;
    }

    protected abstract string MapReasoningEffort(int thinkingBudget);
    protected virtual string GetFileUploadPurpose() => "user_data";

    public async Task<ProviderResponse> CompleteAsync(ProviderRequest request, CancellationToken ct)
    {
        if (UseResponsesApiForRequest(request))
        {
            return await CompleteWithResponsesApiAsync(request, ct).ConfigureAwait(false);
        }

        var body = BuildRequestBody(request, stream: false);
        var jsonBytes = SerializeToUtf8Bytes(body);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {_context.BaseUrl}/v1/chat/completions");

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes, "/v1/chat/completions"),
            ct).ConfigureAwait(false);

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
            EndpointFamily = ModelEndpointFamily.ChatCompletions,
            StopReason = stopReason,
            ToolCalls = toolCalls,
            DiagnosticNote = BuildChatDiagnosticNote(content, stopReason, usage),
        };
    }

    private async Task<ProviderResponse> CompleteWithResponsesApiAsync(ProviderRequest request, CancellationToken ct)
    {
        var uploadedFileIds = await UploadAttachmentsAsync(request, ct).ConfigureAwait(false);

        try
        {
            var body = BuildResponsesRequestBody(request, stream: false, uploadedFileIds);
            var jsonBytes = SerializeToUtf8Bytes(body);

            _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {_context.BaseUrl}/v1/responses");

            using var response = await _context.SendAsync(
                () => CreateHttpRequest(jsonBytes, "/v1/responses"),
                ct).ConfigureAwait(false);

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

            var contentBuilder = new StringBuilder();
            List<ToolCallRequest>? toolCalls = null;

            if (root.TryGetProperty("output", out var output))
            {
                foreach (var item in output.EnumerateArray())
                {
                    if (!item.TryGetProperty("type", out var typeEl))
                    {
                        continue;
                    }

                    switch (typeEl.GetString())
                    {
                        case "message":
                            if (item.TryGetProperty("content", out var contentItems))
                            {
                                foreach (var contentItem in contentItems.EnumerateArray())
                                {
                                    if (contentItem.TryGetProperty("type", out var contentType) &&
                                        contentType.GetString() == "output_text" &&
                                        contentItem.TryGetProperty("text", out var textEl))
                                    {
                                        contentBuilder.Append(textEl.GetString());
                                    }
                                }
                            }
                            break;

                        case "function_call":
                            toolCalls ??= [];
                            toolCalls.Add(new ToolCallRequest
                            {
                                Id = item.GetProperty("call_id").GetString() ?? item.GetProperty("id").GetString()!,
                                ProviderItemId = item.TryGetProperty("id", out var itemId) ? itemId.GetString() : null,
                                Name = item.GetProperty("name").GetString()!,
                                Arguments = item.TryGetProperty("arguments", out var argsEl) ? argsEl.GetString() ?? "" : "",
                            });
                            break;
                    }
                }
            }

            var stopReason = root.TryGetProperty("status", out var statusEl)
                ? statusEl.GetString()
                : null;

            var usage = root.TryGetProperty("usage", out var usageEl)
                ? ParseResponsesUsage(usageEl)
                : new TokenUsage();
            var grounding = ParseResponsesGrounding(root, request);

            return new ProviderResponse
            {
                Content = contentBuilder.ToString(),
                Usage = usage,
                EndpointFamily = ModelEndpointFamily.Responses,
                StopReason = stopReason,
                ToolCalls = toolCalls,
                DiagnosticNote = BuildResponsesDiagnosticNote(contentBuilder.ToString(), stopReason, usage),
                Grounding = grounding,
            };
        }
        finally
        {
            await DeleteUploadedFilesAsync(uploadedFileIds, ct).ConfigureAwait(false);
        }
    }

    public IAsyncEnumerable<ProviderStreamEvent> StreamAsync(
        ProviderRequest request,
        CancellationToken ct)
    {
        if (UseResponsesApiForRequest(request))
            return StreamResponsesApiAsync(request, ct);

        return StreamChatCompletionsAsync(request, ct);
    }

    private async IAsyncEnumerable<ProviderStreamEvent> StreamResponsesApiAsync(
        ProviderRequest request,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var uploadedFileIds = await UploadAttachmentsAsync(request, ct).ConfigureAwait(false);

        try
        {
            var body = BuildResponsesRequestBody(request, stream: true, uploadedFileIds);
            var jsonBytes = SerializeToUtf8Bytes(body);

            _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {_context.BaseUrl}/v1/responses");

            using var response = await _context.SendAsync(
                () => CreateHttpRequest(jsonBytes, "/v1/responses"),
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

            var toolCallArgs = new Dictionary<string, StringBuilder>();
            var emittedTextLength = 0;

            await foreach (var data in SseStreamParser.ParseAsync(stream, ct).ConfigureAwait(false))
            {
                using var doc = JsonDocument.Parse(data);
                var root = doc.RootElement;

                var eventType = root.TryGetProperty("type", out var typeEl) ? typeEl.GetString() : null;

                switch (eventType)
                {
                    case "response.output_text.delta":
                        var delta = root.GetProperty("delta").GetString();
                        if (delta is not null)
                        {
                            emittedTextLength += delta.Length;
                            yield return new ProviderStreamEvent { TextDelta = delta };
                        }
                        break;

                    case "response.function_call_arguments.delta":
                        var callId = root.GetProperty("item_id").GetString()!;
                        var argDelta = root.GetProperty("delta").GetString() ?? "";
                        if (!toolCallArgs.TryGetValue(callId, out var sb))
                        {
                            sb = new StringBuilder();
                            toolCallArgs[callId] = sb;
                        }
                        sb.Append(argDelta);
                        break;

                    case "response.output_item.done":
                        if (root.TryGetProperty("item", out var item) &&
                            item.TryGetProperty("type", out var itemType) &&
                            itemType.GetString() == "function_call")
                        {
                            var providerItemId = item.GetProperty("id").GetString()!;
                            var tcId = item.TryGetProperty("call_id", out var callIdElement)
                                ? callIdElement.GetString() ?? providerItemId
                                : providerItemId;
                            var tcName = item.GetProperty("name").GetString()!;
                            var tcArgs = item.TryGetProperty("arguments", out var argsEl)
                                ? argsEl.GetString() ?? ""
                                : toolCallArgs.TryGetValue(providerItemId, out var accumulated) ? accumulated.ToString() : "";

                            yield return new ProviderStreamEvent
                            {
                                ToolCall = new ToolCallRequest
                                {
                                    Id = tcId,
                                    ProviderItemId = providerItemId,
                                    Name = tcName,
                                    Arguments = tcArgs,
                                },
                            };
                            toolCallArgs.Remove(providerItemId);
                        }
                        break;

                    case "response.completed":
                        if (root.TryGetProperty("response", out var resp))
                        {
                            var completedText = ExtractResponsesOutputText(resp);
                            if (completedText.Length > emittedTextLength)
                            {
                                var remainingText = completedText[emittedTextLength..];
                                if (remainingText.Length > 0)
                                {
                                    yield return new ProviderStreamEvent
                                    {
                                        TextDelta = remainingText,
                                    };
                                }
                            }

                            if (resp.TryGetProperty("usage", out var usageEl))
                            {
                                yield return new ProviderStreamEvent
                                {
                                    Usage = ParseResponsesUsage(usageEl),
                                    Grounding = ParseResponsesGrounding(resp, request),
                                };
                            }
                            else
                            {
                                yield return new ProviderStreamEvent
                                {
                                    Grounding = ParseResponsesGrounding(resp, request),
                                };
                            }
                        }
                        break;
                }
            }
        }
        finally
        {
            await DeleteUploadedFilesAsync(uploadedFileIds, ct).ConfigureAwait(false);
        }
    }

    private async IAsyncEnumerable<ProviderStreamEvent> StreamChatCompletionsAsync(
        ProviderRequest request,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var body = BuildRequestBody(request, stream: true);
        var jsonBytes = SerializeToUtf8Bytes(body);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {_context.BaseUrl}/v1/chat/completions");

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes, "/v1/chat/completions"),
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
                if (usageEl.ValueKind == JsonValueKind.Object)
                {
                    yield return new ProviderStreamEvent
                    {
                        Usage = ParseUsage(usageEl),
                    };
                }
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

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes, "/v1/embeddings"),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Embedding request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        var dto = await InferenceJson.DeserializeAsync(responseStream, InferenceJsonContext.Default.OpenAiEmbeddingResponseDto, ct).ConfigureAwait(false)
            ?? throw new JsonException("OpenAI embedding response was empty.");

        return new ProviderEmbeddingResponse
        {
            Embedding = dto.Data[0].Embedding,
            Usage = new TokenUsage { InputTokens = dto.Usage.PromptTokens },
        };
    }

    public virtual async Task<ProviderBatchEmbeddingResponse> EmbedBatchAsync(
        ProviderBatchEmbeddingRequest request,
        CancellationToken ct)
    {
        var body = BuildBatchEmbeddingBody(request.ModelApiName, request.Inputs, request.Dimensions);
        var jsonBytes = SerializeToUtf8Bytes(body);

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes, "/v1/embeddings"),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Batch embedding request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        var dto = await InferenceJson.DeserializeAsync(responseStream, InferenceJsonContext.Default.OpenAiEmbeddingResponseDto, ct).ConfigureAwait(false)
            ?? throw new JsonException("OpenAI batch embedding response was empty.");
        var embeddings = new ReadOnlyMemory<float>[dto.Data.Length];
        for (var i = 0; i < dto.Data.Length; i++)
        {
            embeddings[i] = dto.Data[i].Embedding;
        }

        return new ProviderBatchEmbeddingResponse
        {
            Embeddings = embeddings,
            Usage = new TokenUsage { InputTokens = dto.Usage.PromptTokens },
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

    private JsonObject BuildResponsesRequestBody(ProviderRequest request, bool stream, IReadOnlyList<string> uploadedAttachmentIds)
    {
        var input = new JsonArray();
        var attachmentIndex = 0;

        if (request.SystemMessage is not null)
            input.Add((JsonNode)new JsonObject { ["role"] = "developer", ["content"] = request.SystemMessage });

        foreach (var msg in request.Messages)
        {
            switch (msg.Role)
            {
                case ChatRole.System:
                    input.Add((JsonNode)new JsonObject { ["role"] = "developer", ["content"] = msg.Content ?? "" });
                    break;
                case ChatRole.User when msg.Attachments is { Count: > 0 }:
                {
                    var parts = new JsonArray();
                    if (!string.IsNullOrEmpty(msg.Content))
                    {
                        parts.Add((JsonNode)new JsonObject { ["type"] = "input_text", ["text"] = msg.Content });
                    }

                    foreach (var attachment in msg.Attachments)
                    {
                        if (attachmentIndex >= uploadedAttachmentIds.Count)
                        {
                            throw new InvalidOperationException("Attachment upload state did not match request attachments.");
                        }

                        parts.Add((JsonNode)new JsonObject
                        {
                            ["type"] = "input_file",
                            ["file_id"] = uploadedAttachmentIds[attachmentIndex++],
                        });
                    }

                    input.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = parts,
                    });
                    break;
                }
                case ChatRole.User:
                    input.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "user",
                        ["content"] = new JsonArray
                        {
                            (JsonNode)new JsonObject { ["type"] = "input_text", ["text"] = msg.Content ?? "" }
                        },
                    });
                    break;
                case ChatRole.Assistant when msg.ToolCalls is { Count: > 0 }:
                    foreach (var tc in msg.ToolCalls)
                    {
                        var functionCall = new JsonObject
                        {
                            ["type"] = "function_call",
                            ["call_id"] = tc.Id,
                            ["name"] = tc.Name,
                            ["arguments"] = tc.Arguments,
                            ["status"] = "completed",
                        };
                        if (!string.IsNullOrWhiteSpace(tc.ProviderItemId))
                            functionCall["id"] = tc.ProviderItemId;
                        input.Add((JsonNode)functionCall);
                    }
                    break;
                case ChatRole.Assistant:
                    break;
                case ChatRole.Tool:
                    input.Add((JsonNode)new JsonObject
                    {
                        ["type"] = "function_call_output",
                        ["call_id"] = msg.ToolCallId,
                        ["output"] = msg.Content ?? "",
                    });
                    break;
            }
        }

        var tools = new JsonArray();

        if (request.Tools is { Count: > 0 })
        {
            foreach (var t in request.Tools)
            {
                tools.Add((JsonNode)new JsonObject
                {
                    ["type"] = "function",
                    ["name"] = t.Name,
                    ["description"] = t.Description,
                    ["parameters"] = JsonNode.Parse(t.ParametersSchema.GetRawText()),
                });
            }
        }

        var grounding = request.Grounding;
        var enableWebSearch = request.EnableWebSearch || grounding?.EnableWebSearch == true;
        var enableXSearch = request.EnableXSearch || grounding?.EnableXSearch == true;

        if (enableWebSearch)
        {
            var webSearch = new JsonObject { ["type"] = "web_search" };
            if (grounding is not null)
            {
                if (IsXai)
                {
                    if (grounding.EnableImageSearch)
                    {
                        webSearch["enable_image_search"] = true;
                    }

                    var filters = BuildXaiSearchFilters(grounding);
                    if (filters.Count > 0)
                    {
                        webSearch["filters"] = filters;
                    }
                }
                else
                {
                    if (grounding.EnableImageSearch)
                    {
                        webSearch["search_content_types"] = new JsonArray("text", "image");
                        webSearch["image_settings"] = new JsonObject
                        {
                            ["max_results"] = Math.Clamp(grounding.MaxSearches, 1, 10),
                            ["caption"] = true,
                        };
                    }

                    if (grounding.AllowedDomains is { Count: > 0 })
                    {
                        webSearch["filters"] = new JsonObject
                        {
                            ["allowed_domains"] = ToJsonArray(grounding.AllowedDomains),
                        };
                    }
                }
            }
            tools.Add((JsonNode)webSearch);
        }
        if (enableXSearch)
            tools.Add((JsonNode)new JsonObject { ["type"] = "x_search" });

        var body = new JsonObject
        {
            ["model"] = request.ModelApiName,
            ["input"] = input,
            ["stream"] = stream,
            ["store"] = false,
        };

        if (tools.Count > 0)
            body["tools"] = tools;

        if (enableWebSearch && !IsXai)
        {
            var include = new JsonArray("web_search_call.action.sources");
            if (grounding?.EnableImageSearch == true)
            {
                include.Add((JsonNode)JsonValue.Create("web_search_call.results")!);
            }
            body["include"] = include;
        }

        if (request.Temperature.HasValue)
            body["temperature"] = request.Temperature.Value;

        if (request.MaxTokens.HasValue)
            body["max_output_tokens"] = request.MaxTokens.Value;

        if (request.ThinkingBudget.HasValue)
        {
            body["reasoning"] = new JsonObject
            {
                ["effort"] = MapReasoningEffort(request.ThinkingBudget.Value),
            };
        }

        if (request.TopP.HasValue)
            body["top_p"] = request.TopP.Value;

        return body;
    }

    private static bool UseResponsesApiForRequest(ProviderRequest request)
    {
        if (request.EnableWebSearch || request.EnableXSearch || request.Grounding is not null)
        {
            return true;
        }

        if (request.Messages.Any(message => message.Attachments is { Count: > 0 }))
        {
            return true;
        }

        return request.PreferredEndpoint == ModelEndpointFamily.Responses;
    }

    private static TokenUsage ParseResponsesUsage(JsonElement usage)
    {
        var inputTokens = usage.TryGetProperty("input_tokens", out var it) ? it.GetInt32() : 0;
        var outputTokens = usage.TryGetProperty("output_tokens", out var ot) ? ot.GetInt32() : 0;

        var cacheReadTokens = 0;
        if (usage.TryGetProperty("input_tokens_details", out var itd) &&
            itd.TryGetProperty("cached_tokens", out var cached))
        {
            cacheReadTokens = cached.GetInt32();
        }

        var thinkingTokens = 0;
        if (usage.TryGetProperty("output_tokens_details", out var otd) &&
            otd.TryGetProperty("reasoning_tokens", out var reasoning))
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

    private bool IsXai => _context.BaseUrl.Contains("api.x.ai", StringComparison.OrdinalIgnoreCase);

    private static JsonObject BuildXaiSearchFilters(GroundingOptions grounding)
    {
        var filters = new JsonObject();
        if (grounding.AllowedDomains is { Count: > 0 })
        {
            filters["allowed_domains"] = ToJsonArray(grounding.AllowedDomains.Take(5));
        }
        else if (grounding.BlockedDomains is { Count: > 0 })
        {
            filters["excluded_domains"] = ToJsonArray(grounding.BlockedDomains.Take(5));
        }
        return filters;
    }

    private static JsonArray ToJsonArray(IEnumerable<string> values)
    {
        var result = new JsonArray();
        foreach (var value in values.Where(value => !string.IsNullOrWhiteSpace(value)))
        {
            result.Add((JsonNode)JsonValue.Create(value.Trim())!);
        }
        return result;
    }

    private static GroundingResult? ParseResponsesGrounding(JsonElement root, ProviderRequest request)
    {
        var sources = new List<GroundingSource>();
        var images = new List<GroundingImage>();
        var queries = new List<string>();
        var webCalls = 0;
        var imageCalls = 0;
        var xCalls = 0;

        if (root.TryGetProperty("output", out var output) && output.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in output.EnumerateArray())
            {
                var type = item.TryGetProperty("type", out var typeEl) ? typeEl.GetString() : null;
                if (type == "message" && item.TryGetProperty("content", out var content))
                {
                    foreach (var block in content.EnumerateArray())
                    {
                        ParseUrlCitations(block, sources);
                    }
                    continue;
                }

                if (type == "x_search_call")
                {
                    xCalls++;
                    ParseSearchAction(item, sources, images, queries);
                    continue;
                }

                if (type == "web_search_call")
                {
                    var beforeImages = images.Count;
                    ParseSearchAction(item, sources, images, queries);
                    var imageSearch = images.Count > beforeImages && request.Grounding?.EnableImageSearch == true;
                    if (item.TryGetProperty("action", out var action) &&
                        action.TryGetProperty("type", out var actionType) &&
                        string.Equals(actionType.GetString(), "image_search", StringComparison.OrdinalIgnoreCase))
                    {
                        imageSearch = true;
                    }
                    if (imageSearch) imageCalls++; else webCalls++;
                }
            }
        }

        if (root.TryGetProperty("citations", out var citations) && citations.ValueKind == JsonValueKind.Array)
        {
            foreach (var citation in citations.EnumerateArray())
            {
                var url = citation.ValueKind == JsonValueKind.String
                    ? citation.GetString()
                    : citation.TryGetProperty("url", out var urlEl) ? urlEl.GetString() : null;
                if (!string.IsNullOrWhiteSpace(url))
                {
                    sources.Add(new GroundingSource { Url = url! });
                }
            }
        }

        if (sources.Count == 0 && images.Count == 0 && queries.Count == 0 && webCalls == 0 && imageCalls == 0 && xCalls == 0)
        {
            return null;
        }

        return new GroundingResult
        {
            Sources = sources.GroupBy(source => source.Url, StringComparer.OrdinalIgnoreCase)
                .Select(group => group.Last())
                .ToArray(),
            Images = images.DistinctBy(image => image.Url, StringComparer.OrdinalIgnoreCase).ToArray(),
            SearchQueries = queries.Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
            Usage = new GroundingUsage
            {
                WebSearchCalls = webCalls,
                ImageSearchCalls = imageCalls,
                XSearchCalls = xCalls,
            },
        };
    }

    private static void ParseUrlCitations(JsonElement block, List<GroundingSource> sources)
    {
        if (!block.TryGetProperty("annotations", out var annotations) || annotations.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        foreach (var annotation in annotations.EnumerateArray())
        {
            if (!annotation.TryGetProperty("type", out var type) || type.GetString() != "url_citation" ||
                !annotation.TryGetProperty("url", out var url) || string.IsNullOrWhiteSpace(url.GetString()))
            {
                continue;
            }

            sources.Add(new GroundingSource
            {
                Url = url.GetString()!,
                Title = annotation.TryGetProperty("title", out var title) ? title.GetString() : null,
                StartIndex = annotation.TryGetProperty("start_index", out var start) ? start.GetInt32() : null,
                EndIndex = annotation.TryGetProperty("end_index", out var end) ? end.GetInt32() : null,
            });
        }
    }

    private static void ParseSearchAction(
        JsonElement item,
        List<GroundingSource> sources,
        List<GroundingImage> images,
        List<string> queries)
    {
        if (item.TryGetProperty("action", out var action))
        {
            if (action.TryGetProperty("query", out var query) && !string.IsNullOrWhiteSpace(query.GetString()))
            {
                queries.Add(query.GetString()!);
            }
            if (action.TryGetProperty("queries", out var queryArray) && queryArray.ValueKind == JsonValueKind.Array)
            {
                foreach (var value in queryArray.EnumerateArray())
                {
                    if (!string.IsNullOrWhiteSpace(value.GetString())) queries.Add(value.GetString()!);
                }
            }
            ParseSourceArray(action, "sources", sources);
        }

        ParseSourceArray(item, "sources", sources);
        if (item.TryGetProperty("results", out var results) && results.ValueKind == JsonValueKind.Array)
        {
            foreach (var result in results.EnumerateArray())
            {
                var imageUrl = result.TryGetProperty("image_url", out var imageUrlEl) ? imageUrlEl.GetString() : null;
                if (!string.IsNullOrWhiteSpace(imageUrl))
                {
                    images.Add(new GroundingImage
                    {
                        Url = imageUrl!,
                        SourceUrl = result.TryGetProperty("source_website_url", out var sourceUrl) ? sourceUrl.GetString() : null,
                        ThumbnailUrl = result.TryGetProperty("thumbnail_url", out var thumbnail) ? thumbnail.GetString() : null,
                        Caption = result.TryGetProperty("caption", out var caption) ? caption.GetString() : null,
                    });
                }
            }
        }
    }

    private static void ParseSourceArray(JsonElement parent, string propertyName, List<GroundingSource> sources)
    {
        if (!parent.TryGetProperty(propertyName, out var array) || array.ValueKind != JsonValueKind.Array) return;
        foreach (var source in array.EnumerateArray())
        {
            var url = source.TryGetProperty("url", out var urlEl) ? urlEl.GetString() : null;
            if (string.IsNullOrWhiteSpace(url)) continue;
            sources.Add(new GroundingSource
            {
                Url = url!,
                Title = source.TryGetProperty("title", out var title) ? title.GetString() : null,
            });
        }
    }

    private static string ExtractResponsesOutputText(JsonElement response)
    {
        var contentBuilder = new StringBuilder();

        if (!response.TryGetProperty("output", out var output) || output.ValueKind != JsonValueKind.Array)
        {
            return "";
        }

        foreach (var item in output.EnumerateArray())
        {
            if (!item.TryGetProperty("type", out var typeEl) ||
                typeEl.ValueKind != JsonValueKind.String ||
                typeEl.GetString() != "message" ||
                !item.TryGetProperty("content", out var contentItems) ||
                contentItems.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            foreach (var contentItem in contentItems.EnumerateArray())
            {
                if (contentItem.TryGetProperty("type", out var contentType) &&
                    contentType.ValueKind == JsonValueKind.String &&
                    contentType.GetString() == "output_text" &&
                    contentItem.TryGetProperty("text", out var textEl) &&
                    textEl.ValueKind == JsonValueKind.String)
                {
                    contentBuilder.Append(textEl.GetString());
                }
            }
        }

        return contentBuilder.ToString();
    }

    protected static ReadOnlyMemory<byte> SerializeToUtf8Bytes(JsonObject body)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using var writer = new Utf8JsonWriter(buffer);
        body.WriteTo(writer);
        writer.Flush();
        return buffer.WrittenMemory;
    }

    private static string? BuildChatDiagnosticNote(string content, string? stopReason, TokenUsage usage)
    {
        if (string.IsNullOrWhiteSpace(content) &&
            string.Equals(stopReason, "length", StringComparison.OrdinalIgnoreCase) &&
            usage.ThinkingTokens > 0)
        {
            return "OpenAI stopped at the output token limit after spending the budget on reasoning.";
        }

        return null;
    }

    private static string? BuildResponsesDiagnosticNote(string content, string? stopReason, TokenUsage usage)
    {
        if (string.IsNullOrWhiteSpace(content) &&
            string.Equals(stopReason, "incomplete", StringComparison.OrdinalIgnoreCase) &&
            usage.ThinkingTokens > 0)
        {
            return "OpenAI Responses completed without visible text after spending output budget on reasoning.";
        }

        return null;
    }

    protected HttpRequestMessage CreateHttpRequest(ReadOnlyMemory<byte> jsonBytes, string path)
    {
        var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_context.BaseUrl}{path}")
        {
            Content = new ReadOnlyMemoryContent(jsonBytes),
        };
        httpRequest.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        httpRequest.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_context.ApiKey}");
        return httpRequest;
    }

    private async Task<List<string>> UploadAttachmentsAsync(ProviderRequest request, CancellationToken ct)
    {
        var uploadedIds = new List<string>();

        foreach (var message in request.Messages)
        {
            if (message.Attachments is not { Count: > 0 })
            {
                continue;
            }

            foreach (var attachment in message.Attachments)
            {
                uploadedIds.Add(await UploadFileAsync(attachment, ct).ConfigureAwait(false));
            }
        }

        return uploadedIds;
    }

    private async Task<string> UploadFileAsync(DocumentAttachment attachment, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent(GetFileUploadPurpose()), "purpose");

        using var fileContent = CreateByteArrayContent(attachment.Content);
        fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse(attachment.MimeType);
        form.Add(fileContent, "file", attachment.FileName);

        using var request = new HttpRequestMessage(HttpMethod.Post, $"{_context.BaseUrl}/v1/files")
        {
            Content = form,
        };
        request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_context.ApiKey}");

        using var response = await _context.SendAsync(
            () =>
            {
                var request = new HttpRequestMessage(HttpMethod.Post, $"{_context.BaseUrl}/v1/files")
                {
                    Content = CreateUploadContent(attachment),
                };
                request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_context.ApiKey}");
                return request;
            },
            ct).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"File upload failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        var dto = await InferenceJson.DeserializeAsync(responseStream, InferenceJsonContext.Default.OpenAiFileUploadResponseDto, ct).ConfigureAwait(false)
            ?? throw new JsonException("OpenAI file upload response was empty.");
        return dto.Id
            ?? throw new InvalidOperationException("Upload response did not include a file id.");
    }

    private async Task DeleteUploadedFilesAsync(IReadOnlyList<string> uploadedFileIds, CancellationToken ct)
    {
        foreach (var fileId in uploadedFileIds)
        {
            try
            {
                using var response = await _context.SendAsync(
                    () =>
                    {
                        var request = new HttpRequestMessage(HttpMethod.Delete, $"{_context.BaseUrl}/v1/files/{WebUtility.UrlEncode(fileId)}");
                        request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_context.ApiKey}");
                        return request;
                    },
                    ct).ConfigureAwait(false);
                _ = response.IsSuccessStatusCode;
            }
            catch
            {
                // Best-effort cleanup only.
            }
        }
    }

    private static ByteArrayContent CreateByteArrayContent(ReadOnlyMemory<byte> content)
    {
        if (MemoryMarshal.TryGetArray(content, out var segment) && segment.Array is not null)
        {
            return new ByteArrayContent(segment.Array, segment.Offset, segment.Count);
        }

        return new ByteArrayContent(content.ToArray());
    }

    private MultipartFormDataContent CreateUploadContent(DocumentAttachment attachment)
    {
        var form = new MultipartFormDataContent();
        form.Add(new StringContent(GetFileUploadPurpose()), "purpose");

        var fileContent = CreateByteArrayContent(attachment.Content);
        fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse(attachment.MimeType);
        form.Add(fileContent, "file", attachment.FileName);

        return form;
    }

    private static TokenUsage ParseUsage(JsonElement usage)
    {
        if (usage.ValueKind != JsonValueKind.Object)
        {
            return new TokenUsage();
        }

        var inputTokens = usage.TryGetProperty("prompt_tokens", out var pt) ? pt.GetInt32() : 0;
        var outputTokens = usage.TryGetProperty("completion_tokens", out var ct) ? ct.GetInt32() : 0;

        var cacheReadTokens = 0;
        if (usage.TryGetProperty("prompt_tokens_details", out var ptd) &&
            ptd.ValueKind == JsonValueKind.Object &&
            ptd.TryGetProperty("cached_tokens", out var cached))
        {
            cacheReadTokens = cached.GetInt32();
        }

        var thinkingTokens = 0;
        if (usage.TryGetProperty("completion_tokens_details", out var ctd) &&
            ctd.ValueKind == JsonValueKind.Object &&
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
