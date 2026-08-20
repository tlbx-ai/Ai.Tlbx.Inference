using System.Buffers;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ai.Tlbx.Inference.Json;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class GoogleProvider : IProvider
{
    private readonly ProviderRequestContext _context;
    private readonly GoogleTokenProvider? _tokenProvider;
    private readonly string? _projectId;
    private readonly string? _location;

    private bool IsVertex => _tokenProvider is not null;

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

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {url}");

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

                        toolCalls.Add(new ToolCallRequest
                        {
                            Id = Guid.NewGuid().ToString("N"),
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

        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Stream request to {url}");

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

                            yield return new ProviderStreamEvent
                            {
                                ToolCall = new ToolCallRequest
                                {
                                    Id = Guid.NewGuid().ToString("N"),
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
        var dto = await InferenceJson.DeserializeAsync(responseStream, InferenceJsonContext.Default.GoogleEmbeddingResponseDto, ct).ConfigureAwait(false)
            ?? throw new JsonException("Google embedding response was empty.");

        return new ProviderEmbeddingResponse
        {
            Embedding = dto.Embedding.Values,
            Usage = new TokenUsage(),
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
        var dto = await InferenceJson.DeserializeAsync(responseStream, InferenceJsonContext.Default.GoogleBatchEmbeddingResponseDto, ct).ConfigureAwait(false)
            ?? throw new JsonException("Google batch embedding response was empty.");
        var embeddings = new ReadOnlyMemory<float>[dto.Embeddings.Length];
        for (var i = 0; i < dto.Embeddings.Length; i++)
        {
            embeddings[i] = dto.Embeddings[i].Values;
        }

        return new ProviderBatchEmbeddingResponse
        {
            Embeddings = embeddings,
            Usage = new TokenUsage(),
        };
    }

    public async Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
    {
        var generationConfig = new JsonObject
        {
            ["responseModalities"] = new JsonArray("IMAGE"),
        };

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

        var jsonBytes = SerializeToUtf8Bytes(body);
        var url = BuildUrl(request.ModelApiName, "generateContent");

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
            ?? throw new InvalidOperationException("Google image generation response did not include image data.");

        return Convert.FromBase64String(inlineData);
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
                        var part = new JsonObject
                        {
                            ["functionCall"] = new JsonObject
                            {
                                ["name"] = tc.Name,
                                ["args"] = JsonNode.Parse(tc.Arguments),
                            },
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

                    contents.Add((JsonNode)new JsonObject
                    {
                        ["role"] = "user",
                        ["parts"] = new JsonArray
                        {
                            (JsonNode)new JsonObject
                            {
                                ["functionResponse"] = new JsonObject
                                {
                                    ["name"] = msg.ToolCallId ?? "",
                                    ["response"] = new JsonObject { ["result"] = resultValue },
                                },
                            },
                        },
                    });
                    break;
                }
            }
        }

        body["contents"] = contents;

        var generationConfig = new JsonObject();

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

        if (request.ThinkingBudget.HasValue)
        {
            generationConfig["thinkingConfig"] = new JsonObject
            {
                ["thinkingBudget"] = request.ThinkingBudget.Value,
            };
        }

        if (request.JsonSchema is not null)
        {
            generationConfig["responseMimeType"] = "application/json";
            generationConfig["responseSchema"] = JsonNode.Parse(request.JsonSchema);
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
                    ["parameters"] = JsonNode.Parse(t.ParametersSchema.GetRawText()),
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

    private string BuildUrl(string model, string method)
    {
        if (IsVertex)
        {
            return $"https://{_location}-aiplatform.googleapis.com/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{model}:{method}";
        }

        return $"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}?key={_context.ApiKey}";
    }

    private string BuildStreamUrl(string model)
    {
        if (IsVertex)
        {
            return $"https://{_location}-aiplatform.googleapis.com/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{model}:streamGenerateContent?alt=sse";
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
        if (string.IsNullOrWhiteSpace(content) &&
            string.Equals(stopReason, "MAX_TOKENS", StringComparison.OrdinalIgnoreCase) &&
            usage.ThinkingTokens > 0)
        {
            return "Google stopped at the output token limit after spending tokens on thinking before emitting visible text.";
        }

        return null;
    }
}
