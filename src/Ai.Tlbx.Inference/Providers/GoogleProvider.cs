using System.Buffers;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class GoogleProvider : IProvider
{
    private readonly ProviderRequestContext _context;
    private readonly GoogleTokenProvider? _tokenProvider;
    private readonly string? _projectId;
    private readonly string? _location;

    private bool IsVertex => _tokenProvider is not null;
    private string VertexHost => _location == "global" ? "aiplatform.googleapis.com" : $"{_location}-aiplatform.googleapis.com";

    public GoogleProvider(
        ProviderRequestContext context,
        GoogleTokenProvider? tokenProvider = null,
        string? projectId = null,
        string? location = null)
    {
        _context = context;
        _tokenProvider = tokenProvider;
        _projectId = projectId;
        _location = location;
    }

    public async Task<ProviderResponse> CompleteAsync(ProviderRequest request, CancellationToken ct)
    {
        var body = BuildRequestBody(request);
        var jsonBytes = SerializeToUtf8Bytes(body);
        var url = BuildUrl(request.ModelApiName, "generateContent");

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {url.Split('?')[0]}");

        using var response = await _context.SendAsync(
            token => CreateHttpRequestAsync(jsonBytes, url, token),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Google API request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;

        var contentBuilder = new StringBuilder();
        List<ToolCallRequest>? toolCalls = null;

        if (root.TryGetProperty("candidates", out var candidates) && candidates.GetArrayLength() > 0)
        {
            var candidate = candidates[0];
            var stopReason = candidate.TryGetProperty("finishReason", out var finishEl)
                ? finishEl.GetString()
                : null;
            if (candidate.TryGetProperty("content", out var content) &&
                content.TryGetProperty("parts", out var parts))
            {
                foreach (var part in parts.EnumerateArray())
                {
                    if (part.TryGetProperty("thought", out var thought) && thought.ValueKind == JsonValueKind.True)
                    {
                        continue;
                    }
                    if (part.TryGetProperty("text", out var textEl))
                    {
                        contentBuilder.Append(textEl.GetString());
                    }
                    else if (part.TryGetProperty("functionCall", out var fnCall))
                    {
                        toolCalls ??= [];
                        var name = fnCall.GetProperty("name").GetString()!;
                        var args = fnCall.TryGetProperty("args", out var argsEl)
                            ? argsEl.GetRawText()
                            : "{}";
                        var thoughtSignature = ExtractThoughtSignature(part, fnCall);
                        var nativeId = ReadFunctionCallId(fnCall);

                        toolCalls.Add(new ToolCallRequest
                        {
                            Id = nativeId ?? Guid.NewGuid().ToString("N"),
                            ProviderItemId = nativeId,
                            Name = name,
                            Arguments = args,
                            ThoughtSignature = thoughtSignature,
                        });
                    }
                }
            }
        }

        var usage = ParseUsage(root);
        var grounding = ParseGrounding(root);

        return new ProviderResponse
        {
            Content = contentBuilder.ToString(),
            Usage = usage,
            EndpointFamily = ModelEndpointFamily.GoogleGenerateContent,
            StopReason = root.TryGetProperty("candidates", out var rootCandidates) && rootCandidates.GetArrayLength() > 0 &&
                         rootCandidates[0].TryGetProperty("finishReason", out var stopEl)
                ? stopEl.GetString()
                : null,
            ToolCalls = toolCalls,
            DiagnosticNote = BuildDiagnosticNote(
                contentBuilder.ToString(),
                root.TryGetProperty("candidates", out var candidatesForNote) && candidatesForNote.GetArrayLength() > 0 &&
                candidatesForNote[0].TryGetProperty("finishReason", out var stopForNote)
                    ? stopForNote.GetString()
                    : null,
                usage),
            Grounding = grounding,
        };
    }

    public async IAsyncEnumerable<ProviderStreamEvent> StreamAsync(
        ProviderRequest request,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var body = BuildRequestBody(request);
        var jsonBytes = SerializeToUtf8Bytes(body);

        var url = BuildStreamUrl(request.ModelApiName);

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {url.Split('?')[0]}");

        using var response = await _context.SendAsync(
            token => CreateHttpRequestAsync(jsonBytes, url, token),
            HttpCompletionOption.ResponseHeadersRead,
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Google stream request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);

        await foreach (var data in SseStreamParser.ParseAsync(stream, ct).ConfigureAwait(false))
        {
            using var doc = JsonDocument.Parse(data);
            var root = doc.RootElement;

            if (root.TryGetProperty("candidates", out var candidates) && candidates.GetArrayLength() > 0)
            {
                var candidate = candidates[0];
                if (candidate.TryGetProperty("content", out var content) &&
                    content.TryGetProperty("parts", out var parts))
                {
                    foreach (var part in parts.EnumerateArray())
                    {
                        if (part.TryGetProperty("thought", out var thought) && thought.ValueKind == JsonValueKind.True)
                        {
                            continue;
                        }
                        if (part.TryGetProperty("text", out var textEl))
                        {
                            var text = textEl.GetString();
                            if (text is not null)
                            {
                                yield return new ProviderStreamEvent { TextDelta = text };
                            }
                        }
                        else if (part.TryGetProperty("functionCall", out var fnCall))
                        {
                            var name = fnCall.GetProperty("name").GetString()!;
                            var args = fnCall.TryGetProperty("args", out var argsEl)
                                ? argsEl.GetRawText()
                                : "{}";
                            var thoughtSignature = ExtractThoughtSignature(part, fnCall);
                            var nativeId = ReadFunctionCallId(fnCall);

                            yield return new ProviderStreamEvent
                            {
                                ToolCall = new ToolCallRequest
                                {
                                    Id = nativeId ?? Guid.NewGuid().ToString("N"),
                                    ProviderItemId = nativeId,
                                    Name = name,
                                    Arguments = args,
                                    ThoughtSignature = thoughtSignature,
                                },
                            };
                        }
                    }
                }
            }

            if (root.TryGetProperty("usageMetadata", out var usageMeta))
            {
                yield return new ProviderStreamEvent
                {
                    Usage = ParseUsageMetadata(usageMeta),
                    Grounding = ParseGrounding(root),
                };
            }
        }
    }

    public async Task<ProviderEmbeddingResponse> EmbedAsync(ProviderEmbeddingRequest request, CancellationToken ct)
    {
        var body = new JsonObject
        {
            ["model"] = $"models/{request.ModelApiName}",
            ["content"] = new JsonObject
            {
                ["parts"] = new JsonArray { (JsonNode)new JsonObject { ["text"] = request.Input } },
            },
        };

        if (request.Dimensions.HasValue)
        {
            body["outputDimensionality"] = request.Dimensions.Value;
        }

        var jsonBytes = SerializeToUtf8Bytes(body);
        var url = BuildUrl(request.ModelApiName, "embedContent");

        using var response = await _context.SendAsync(
            token => CreateHttpRequestAsync(jsonBytes, url, token),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Google embedding request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;

        return new ProviderEmbeddingResponse
        {
            Embedding = ParseEmbeddingValues(root.GetProperty("embedding").GetProperty("values")),
            Usage = ReadEmbeddingUsage(root),
        };
    }

    public async Task<ProviderBatchEmbeddingResponse> EmbedBatchAsync(
        ProviderBatchEmbeddingRequest request,
        CancellationToken ct)
    {
        var requestsArray = new JsonArray();
        foreach (var input in request.Inputs)
        {
            var req = new JsonObject
            {
                ["model"] = $"models/{request.ModelApiName}",
                ["content"] = new JsonObject
                {
                    ["parts"] = new JsonArray { (JsonNode)new JsonObject { ["text"] = input } },
                },
            };

            if (request.Dimensions.HasValue)
            {
                req["outputDimensionality"] = request.Dimensions.Value;
            }

            requestsArray.Add((JsonNode)req);
        }

        var body = new JsonObject { ["requests"] = requestsArray };
        var jsonBytes = SerializeToUtf8Bytes(body);
        var url = BuildUrl(request.ModelApiName, "batchEmbedContents");

        using var response = await _context.SendAsync(
            token => CreateHttpRequestAsync(jsonBytes, url, token),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Google batch embedding request failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var root = doc.RootElement;
        var embeddingArray = root.GetProperty("embeddings");
        var embeddings = new ReadOnlyMemory<float>[embeddingArray.GetArrayLength()];
        var itemUsage = new TokenUsage();
        for (var i = 0; i < embeddings.Length; i++)
        {
            embeddings[i] = ParseEmbeddingValues(embeddingArray[i].GetProperty("values"));
            itemUsage += ReadEmbeddingUsage(embeddingArray[i]);
        }

        return new ProviderBatchEmbeddingResponse
        {
            Embeddings = embeddings,
            // A batch total already includes its items. Only sum item metadata when no total was reported.
            Usage = HasEmbeddingUsage(root) ? ReadEmbeddingUsage(root) : itemUsage,
        };
    }

    private static bool HasEmbeddingUsage(JsonElement root)
        => root.TryGetProperty("usageMetadata", out var metadata)
            && metadata.ValueKind == JsonValueKind.Object
            && metadata.TryGetProperty("promptTokenCount", out _);

    private static TokenUsage ReadEmbeddingUsage(JsonElement root)
        => new()
        {
            InputTokens = HasEmbeddingUsage(root)
                ? root.GetProperty("usageMetadata").GetProperty("promptTokenCount").GetInt32()
                : 0,
        };

    public Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
    {
        var generationConfig = BuildImageGenerationConfig(request.ModelApiName, request.Size, request.Quality);

        var body = new JsonObject
        {
            ["contents"] = new JsonArray
            {
                (JsonNode)new JsonObject
                {
                    ["parts"] = new JsonArray { (JsonNode)new JsonObject { ["text"] = request.Prompt } },
                },
            },
            ["generationConfig"] = generationConfig,
        };

        return SendImageRequestAsync(request.ModelApiName, body, ct);
    }

    public Task<byte[]> EditImageAsync(ProviderImageEditRequest request, CancellationToken ct)
    {
        var generationConfig = BuildImageGenerationConfig(request.ModelApiName, request.Size, request.Quality);
        if (request.Images.Count == 0)
        {
            throw new ArgumentException("At least one reference image is required for image editing.", nameof(request));
        }
        var parts = new JsonArray();
        foreach (var image in request.Images)
        {
            parts.Add((JsonNode)new JsonObject
            {
                ["inlineData"] = new JsonObject
                {
                    ["mimeType"] = image.MimeType,
                    ["data"] = Convert.ToBase64String(image.Content.Span),
                },
            });
        }
        parts.Add((JsonNode)new JsonObject { ["text"] = request.Prompt });
        var body = new JsonObject
        {
            ["contents"] = new JsonArray
            {
                (JsonNode)new JsonObject { ["role"] = "user", ["parts"] = parts },
            },
            ["generationConfig"] = generationConfig,
        };
        return SendImageRequestAsync(request.ModelApiName, body, ct);
    }

    private static JsonObject BuildImageGenerationConfig(string model, string? size, string? quality)
    {
        if (quality is not null)
        {
            throw new ArgumentException("Google image models do not support the Quality parameter. Use Size with a supported resolution instead.", nameof(quality));
        }
        var config = new JsonObject { ["responseModalities"] = new JsonArray("IMAGE") };
        if (size is not null)
        {
            if (model == "gemini-2.5-flash-image" || size is not ("1K" or "2K" or "4K"))
            {
                throw new ArgumentException("Google image Size supports 1K, 2K, or 4K on Gemini 3 image models and Nano Banana 2.1. Omit Size for Gemini 2.5 Flash Image.", nameof(size));
            }
            config["imageConfig"] = new JsonObject { ["imageSize"] = size };
        }
        return config;
    }

    private async Task<byte[]> SendImageRequestAsync(string model, JsonObject body, CancellationToken ct)
    {
        var jsonBytes = SerializeToUtf8Bytes(body);
        var url = BuildUrl(model, "generateContent");

        using var response = await _context.SendAsync(
            token => CreateHttpRequestAsync(jsonBytes, url, token),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Google image generation failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var doc = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);
        var inlineData = FindInlineImageData(doc.RootElement)
            ?? throw new InvalidOperationException(DescribeMissingImage(doc.RootElement));

        return Convert.FromBase64String(inlineData);
    }

    private static string DescribeMissingImage(JsonElement root)
    {
        var details = new List<string>();
        if (root.TryGetProperty("promptFeedback", out var feedback)
            && feedback.TryGetProperty("blockReason", out var blockReason)
            && blockReason.ValueKind == JsonValueKind.String)
        {
            var reason = blockReason.GetString()!;
            details.Add($"Prompt block reason: {reason[..Math.Min(reason.Length, 128)]}.");
        }
        var text = new StringBuilder();
        if (root.TryGetProperty("candidates", out var candidates) && candidates.ValueKind == JsonValueKind.Array)
        {
            foreach (var candidate in candidates.EnumerateArray())
            {
                if (candidate.TryGetProperty("finishReason", out var finishReason) && finishReason.ValueKind == JsonValueKind.String)
                {
                    var reason = finishReason.GetString()!;
                    details.Add($"Finish reason: {reason[..Math.Min(reason.Length, 128)]}.");
                }
                if (!candidate.TryGetProperty("content", out var content)
                    || !content.TryGetProperty("parts", out var parts) || parts.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }
                foreach (var part in parts.EnumerateArray())
                {
                    if (part.TryGetProperty("thought", out var thought) && thought.ValueKind == JsonValueKind.True)
                    {
                        continue;
                    }
                    if (text.Length < 400 && part.TryGetProperty("text", out var partText) && partText.ValueKind == JsonValueKind.String)
                    {
                        var value = partText.GetString()!;
                        text.Append(value.AsSpan(0, Math.Min(value.Length, 400 - text.Length)));
                    }
                }
            }
        }
        if (text.Length > 0) details.Add($"Provider text: {text}");
        return "Google image generation response did not include image data."
            + (details.Count == 0 ? "" : $" {string.Join(" ", details)}");
    }

    private JsonObject BuildRequestBody(ProviderRequest request)
    {
        var body = new JsonObject();

        if (request.SystemMessage is not null)
        {
            body["system_instruction"] = new JsonObject
            {
                ["parts"] = new JsonArray { (JsonNode)new JsonObject { ["text"] = request.SystemMessage } },
            };
        }

        var contents = new JsonArray();
        var toolCallsById = new Dictionary<string, ToolCallRequest>(StringComparer.Ordinal);

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
                        parts.Add((JsonNode)new JsonObject
                        {
                            ["inline_data"] = new JsonObject
                            {
                                ["mime_type"] = attachment.MimeType,
                                ["data"] = base64,
                            },
                        });
                    }

                    if (!string.IsNullOrEmpty(msg.Content))
                    {
                        parts.Add((JsonNode)new JsonObject { ["text"] = msg.Content });
                    }

                    contents.Add((JsonNode)new JsonObject { ["role"] = "user", ["parts"] = parts });
                    break;
                }

                case ChatRole.User:
                    contents.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "user",
                        ["parts"] = new JsonArray { (JsonNode)new JsonObject { ["text"] = msg.Content ?? "" } },
                    });
                    break;

                case ChatRole.Assistant when msg.ToolCalls is { Count: > 0 }:
                {
                    var parts = new JsonArray();
                    if (!string.IsNullOrEmpty(msg.Content))
                    {
                        parts.Add((JsonNode)new JsonObject { ["text"] = msg.Content });
                    }
                    foreach (var tc in msg.ToolCalls)
                    {
                        toolCallsById[tc.Id] = tc;
                        var functionCall = new JsonObject
                        {
                            ["name"] = tc.Name,
                            ["args"] = JsonNode.Parse(tc.Arguments),
                        };
                        if (!string.IsNullOrWhiteSpace(tc.ProviderItemId))
                        {
                            functionCall["id"] = tc.ProviderItemId;
                        }
                        var part = new JsonObject
                        {
                            ["functionCall"] = functionCall,
                        };
                        if (!string.IsNullOrWhiteSpace(tc.ThoughtSignature))
                        {
                            part["thoughtSignature"] = tc.ThoughtSignature;
                        }

                        parts.Add((JsonNode)part);
                    }
                    contents.Add((JsonNode)new JsonObject { ["role"] = "model", ["parts"] = parts });
                    break;
                }

                case ChatRole.Assistant:
                    contents.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "model",
                        ["parts"] = new JsonArray { (JsonNode)new JsonObject { ["text"] = msg.Content ?? "" } },
                    });
                    break;

                case ChatRole.Tool:
                {
                    JsonNode resultValue;
                    try
                    {
                        resultValue = JsonNode.Parse(msg.Content ?? "{}") ?? JsonValue.Create(msg.Content ?? "");
                    }
                    catch (JsonException)
                    {
                        resultValue = JsonValue.Create(msg.Content ?? "");
                    }

                    toolCallsById.TryGetValue(msg.ToolCallId ?? "", out var matchingCall);
                    var functionResponse = new JsonObject
                    {
                        ["name"] = matchingCall?.Name ?? msg.ToolCallId ?? "",
                        ["response"] = new JsonObject { ["result"] = resultValue },
                    };
                    if (!string.IsNullOrWhiteSpace(matchingCall?.ProviderItemId))
                    {
                        functionResponse["id"] = matchingCall.ProviderItemId;
                    }
                    contents.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "user",
                        ["parts"] = new JsonArray
                        {
                            (JsonNode)new JsonObject
                            {
                                ["functionResponse"] = functionResponse,
                            },
                        },
                    });
                    break;
                }
            }
        }

        body["contents"] = contents;

        var generationConfig = new JsonObject();

        var isArgon = request.ModelApiName.StartsWith("gemini-4-", StringComparison.Ordinal);
        if (isArgon && (request.Temperature.HasValue || request.TopP.HasValue))
        {
            throw new ArgumentException("Gemini 4 Argon does not support temperature or top-p sampling controls.", nameof(request));
        }
        if (request.Temperature.HasValue)
        {
            generationConfig["temperature"] = request.Temperature.Value;
        }

        if (request.MaxTokens.HasValue)
        {
            generationConfig["maxOutputTokens"] = request.MaxTokens.Value;
        }

        if (request.TopP.HasValue)
        {
            generationConfig["topP"] = request.TopP.Value;
        }

        if (request.StopSequences is { Count: > 0 })
        {
            var stopArray = new JsonArray();
            foreach (var seq in request.StopSequences)
            {
                stopArray.Add((JsonNode)JsonValue.Create(seq)!);
            }
            generationConfig["stopSequences"] = stopArray;
        }

        if (request.ReasoningEffort is not null || request.ThinkingBudget.HasValue)
        {
            if (request.ModelApiName.StartsWith("gemini-3", StringComparison.Ordinal) || isArgon)
            {
                var level = request.ReasoningEffort ?? MapThinkingBudget(request.ThinkingBudget!.Value);
                if (level is not ("low" or "medium" or "high"))
                {
                    throw new ArgumentException("Gemini thinking levels supported by this client are low, medium, and high.", nameof(request));
                }
                generationConfig["thinkingConfig"] = new JsonObject { ["thinkingLevel"] = level.ToUpperInvariant() };
            }
            else
            {
                if (request.ReasoningEffort is not null)
                {
                    throw new ArgumentException("Use ThinkingBudget for Gemini models older than Gemini 3.", nameof(request));
                }
                generationConfig["thinkingConfig"] = new JsonObject { ["thinkingBudget"] = request.ThinkingBudget!.Value };
            }
        }

        if (request.JsonSchema is not null)
        {
            generationConfig["responseMimeType"] = "application/json";
            generationConfig["responseJsonSchema"] = JsonNode.Parse(request.JsonSchema);
        }

        if (generationConfig.Count > 0)
        {
            body["generationConfig"] = generationConfig;
        }

        var tools = new JsonArray();
        if (request.Tools is { Count: > 0 })
        {
            var declarations = new JsonArray();
            foreach (var t in request.Tools)
            {
                declarations.Add((JsonNode)new JsonObject
                {
                    ["name"] = t.Name,
                    ["description"] = t.Description,
                    ["parametersJsonSchema"] = JsonNode.Parse(t.ParametersSchema.GetRawText()),
                });
            }
            tools.Add((JsonNode)new JsonObject { ["function_declarations"] = declarations });
        }

        if (request.EnableWebSearch || request.Grounding?.EnableWebSearch == true)
        {
            tools.Add((JsonNode)new JsonObject { ["googleSearch"] = new JsonObject() });
        }

        if (tools.Count > 0)
        {
            body["tools"] = tools;
        }

        return body;
    }

    private static string MapThinkingBudget(int budget) => budget switch
    {
        <= 1024 => "low",
        <= 8192 => "medium",
        _ => "high",
    };

    private static string? ReadFunctionCallId(JsonElement functionCall)
        => functionCall.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.String
            && !string.IsNullOrWhiteSpace(id.GetString()) ? id.GetString() : null;

    private string BuildUrl(string model, string method)
    {
        if (IsVertex)
        {
            return $"https://{VertexHost}/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{model}:{method}";
        }

        return $"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}?key={_context.ApiKey}";
    }

    private string BuildStreamUrl(string model)
    {
        if (IsVertex)
        {
            return $"https://{VertexHost}/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{model}:streamGenerateContent?alt=sse";
        }

        return $"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={_context.ApiKey}";
    }

    private async Task<HttpRequestMessage> CreateHttpRequestAsync(
        ReadOnlyMemory<byte> jsonBytes,
        string url,
        CancellationToken ct)
    {
        var content = new ReadOnlyMemoryContent(jsonBytes);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };

        var httpRequest = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = content,
        };

        if (IsVertex)
        {
            var token = await _tokenProvider!.GetAccessTokenAsync(ct).ConfigureAwait(false);
            httpRequest.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        return httpRequest;
    }

    private static ReadOnlyMemory<byte> SerializeToUtf8Bytes(JsonObject body)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using var writer = new Utf8JsonWriter(buffer);
        body.WriteTo(writer);
        writer.Flush();
        return buffer.WrittenMemory;
    }

    private static TokenUsage ParseUsage(JsonElement root)
    {
        if (!root.TryGetProperty("usageMetadata", out var usageMeta))
        {
            return new TokenUsage();
        }

        return ParseUsageMetadata(usageMeta);
    }

    private static TokenUsage ParseUsageMetadata(JsonElement usageMeta)
    {
        var inputTokens = usageMeta.TryGetProperty("promptTokenCount", out var pt) ? pt.GetInt32() : 0;
        var outputTokens = usageMeta.TryGetProperty("candidatesTokenCount", out var ct) ? ct.GetInt32() : 0;
        var cacheRead = usageMeta.TryGetProperty("cachedContentTokenCount", out var cc) ? cc.GetInt32() : 0;
        var thinkingTokens = usageMeta.TryGetProperty("thoughtsTokenCount", out var tt) ? tt.GetInt32() : 0;

        return new TokenUsage
        {
            InputTokens = inputTokens,
            OutputTokens = outputTokens + thinkingTokens,
            CacheReadTokens = cacheRead,
            ThinkingTokens = thinkingTokens,
        };
    }

    private static float[] ParseEmbeddingValues(JsonElement valuesEl)
    {
        var arr = new float[valuesEl.GetArrayLength()];
        var i = 0;
        foreach (var val in valuesEl.EnumerateArray())
        {
            arr[i++] = val.GetSingle();
        }
        return arr;
    }

    private static string? FindInlineImageData(JsonElement root)
    {
        if (!root.TryGetProperty("candidates", out var candidates))
        {
            return null;
        }

        foreach (var candidate in candidates.EnumerateArray())
        {
            if (!candidate.TryGetProperty("content", out var content) ||
                !content.TryGetProperty("parts", out var parts))
            {
                continue;
            }

            foreach (var part in parts.EnumerateArray())
            {
                if (part.TryGetProperty("thought", out var thought) && thought.ValueKind == JsonValueKind.True)
                {
                    continue;
                }
                if (TryGetInlineData(part, out var inlineData) &&
                    inlineData.TryGetProperty("data", out var data) &&
                    !string.IsNullOrWhiteSpace(data.GetString()))
                {
                    return data.GetString();
                }
            }
        }

        return null;
    }

    private static bool TryGetInlineData(JsonElement part, out JsonElement inlineData)
    {
        if (part.TryGetProperty("inlineData", out inlineData))
        {
            return true;
        }

        if (part.TryGetProperty("inline_data", out inlineData))
        {
            return true;
        }

        inlineData = default;
        return false;
    }

    private static GroundingResult? ParseGrounding(JsonElement root)
    {
        if (!root.TryGetProperty("candidates", out var candidates) || candidates.GetArrayLength() == 0 ||
            !candidates[0].TryGetProperty("groundingMetadata", out var metadata))
        {
            return null;
        }

        var queries = new List<string>();
        if (metadata.TryGetProperty("webSearchQueries", out var queryArray) && queryArray.ValueKind == JsonValueKind.Array)
        {
            foreach (var query in queryArray.EnumerateArray())
            {
                if (!string.IsNullOrWhiteSpace(query.GetString())) queries.Add(query.GetString()!);
            }
        }

        var sources = new List<GroundingSource>();
        if (metadata.TryGetProperty("groundingChunks", out var chunks) && chunks.ValueKind == JsonValueKind.Array)
        {
            foreach (var chunk in chunks.EnumerateArray())
            {
                if (!chunk.TryGetProperty("web", out var web) ||
                    !web.TryGetProperty("uri", out var uri) || string.IsNullOrWhiteSpace(uri.GetString()))
                {
                    continue;
                }

                sources.Add(new GroundingSource
                {
                    Url = uri.GetString()!,
                    Title = web.TryGetProperty("title", out var title) ? title.GetString() : null,
                });
            }
        }

        var calls = ParseBilledGoogleSearchCalls(root);
        if (calls == 0)
        {
            calls = queries.Where(query => !string.IsNullOrWhiteSpace(query))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Count();
        }

        if (sources.Count == 0 && queries.Count == 0 && calls == 0) return null;
        return new GroundingResult
        {
            Sources = sources.DistinctBy(source => source.Url, StringComparer.OrdinalIgnoreCase).ToArray(),
            SearchQueries = queries.Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
            Usage = new GroundingUsage { WebSearchCalls = calls },
        };
    }

    private static int ParseBilledGoogleSearchCalls(JsonElement root)
    {
        if (!root.TryGetProperty("usageMetadata", out var usage) ||
            !usage.TryGetProperty("billedToolCalls", out var billed) || billed.ValueKind != JsonValueKind.Array)
        {
            return 0;
        }

        var count = 0;
        foreach (var entry in billed.EnumerateArray())
        {
            var tool = entry.TryGetProperty("tool", out var toolEl) ? toolEl.GetString() : null;
            if (string.Equals(tool, "GOOGLE_SEARCH_RETRIEVAL", StringComparison.OrdinalIgnoreCase) &&
                entry.TryGetProperty("successfulToolCallCount", out var value))
            {
                count += value.GetInt32();
            }
        }
        return count;
    }

    private static string? ExtractThoughtSignature(JsonElement part, JsonElement functionCall)
    {
        if (part.TryGetProperty("thoughtSignature", out var camelPart) && camelPart.ValueKind == JsonValueKind.String)
        {
            return camelPart.GetString();
        }

        if (part.TryGetProperty("thought_signature", out var snakePart) && snakePart.ValueKind == JsonValueKind.String)
        {
            return snakePart.GetString();
        }

        if (functionCall.TryGetProperty("thoughtSignature", out var camelCall) && camelCall.ValueKind == JsonValueKind.String)
        {
            return camelCall.GetString();
        }

        if (functionCall.TryGetProperty("thought_signature", out var snakeCall) && snakeCall.ValueKind == JsonValueKind.String)
        {
            return snakeCall.GetString();
        }

        return null;
    }

    private static string? BuildDiagnosticNote(string content, string? stopReason, TokenUsage usage)
    {
        if (string.Equals(stopReason, "CONTINUATION", StringComparison.OrdinalIgnoreCase))
        {
            return "Google paused at a per-request output limit. Automatic long-decode continuation is not supported by this client.";
        }
        if (string.IsNullOrWhiteSpace(content) &&
            string.Equals(stopReason, "MAX_TOKENS", StringComparison.OrdinalIgnoreCase) &&
            usage.ThinkingTokens > 0)
        {
            return "Google stopped at the output token limit after spending tokens on thinking before emitting visible text.";
        }

        return null;
    }
}
