using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Json;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Resilience;
using Ai.Tlbx.Inference.Schema;
using Polly;

namespace Ai.Tlbx.Inference;

public sealed class AiInferenceClient : IAiInferenceClient
{
    private readonly Dictionary<ProviderType, IProvider> _providers = new();
    private readonly ResiliencePipeline<HttpResponseMessage> _resiliencePipeline;
    private readonly Action<InferenceLogLevel, string>? _log;

    public AiInferenceClient(HttpClient httpClient, AiInferenceOptions options)
    {
        _log = options.LogAction;
        _resiliencePipeline = options.CustomRetryPolicy ?? RetryPolicyFactory.CreateDefault(_log);

        foreach (var (providerType, creds) in options.Providers)
        {
            var provider = CreateProvider(providerType, creds, httpClient, _log);
            _providers[providerType] = provider;
        }
    }

    public async Task<CompletionResponse<string>> CompleteAsync(CompletionRequest request, CancellationToken ct = default)
    {
        ValidateGroundingRequest(request);
        var descriptor = AiModelCatalog.Get(request.Model);
        var provider = GetProvider(descriptor.Provider);
        var providerRequest = BuildProviderRequest(request);
        var response = await provider.CompleteAsync(providerRequest, ct).ConfigureAwait(false);

        return new CompletionResponse<string>
        {
            Content = response.Content,
            Usage = response.Usage,
            Model = request.Model,
            StopReason = response.StopReason,
            Diagnostics = BuildDiagnostics(descriptor, response),
            Grounding = response.Grounding,
        };
    }

    public async Task<CompletionResponse<T>> CompleteAsync<T>(CompletionRequest request, JsonTypeInfo<T> jsonTypeInfo, CancellationToken ct = default)
    {
        ValidateGroundingRequest(request);
        var schema = request.JsonSchema
            ?? throw new InvalidOperationException("CompletionRequest.JsonSchema must be provided for AOT-compatible structured output.");
        var descriptor = AiModelCatalog.Get(request.Model);
        var provider = GetProvider(descriptor.Provider);
        var providerRequest = BuildProviderRequest(request, schema);
        var response = await provider.CompleteAsync(providerRequest, ct).ConfigureAwait(false);

        var content = JsonSerializer.Deserialize(response.Content, jsonTypeInfo)!;

        return new CompletionResponse<T>
        {
            Content = content,
            Usage = response.Usage,
            Model = request.Model,
            StopReason = response.StopReason,
            Diagnostics = BuildDiagnostics(descriptor, response),
            Grounding = response.Grounding,
        };
    }

    public async IAsyncEnumerable<string> StreamAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ValidateGroundingRequest(request);
        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = BuildProviderRequest(request);

        await foreach (var e in provider.StreamAsync(providerRequest, ct).ConfigureAwait(false))
        {
            if (e.TextDelta is not null)
            {
                foreach (var normalizedChunk in SplitStreamingText(e.TextDelta))
                {
                    yield return normalizedChunk;
                }
            }
        }
    }

    private static IEnumerable<string> SplitStreamingText(string text)
    {
        const int preferredChunkLength = 160;

        if (text.Length <= preferredChunkLength)
        {
            yield return text;
            yield break;
        }

        var start = 0;
        while (start < text.Length)
        {
            var remaining = text.Length - start;
            if (remaining <= preferredChunkLength)
            {
                yield return text[start..];
                yield break;
            }

            var end = start + preferredChunkLength;
            while (end > start && !char.IsWhiteSpace(text[end - 1]))
            {
                end--;
            }

            if (end == start)
            {
                end = start + preferredChunkLength;
            }

            yield return text[start..end];
            start = end;
        }
    }

    public async Task<ToolExecutionResponse<string>> CompleteWithToolsAsync(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        int maxIterations = 20,
        CancellationToken ct = default)
    {
        ValidateGroundingRequest(request);
        var descriptor = AiModelCatalog.Get(request.Model);
        var provider = GetProvider(descriptor.Provider);
        var messages = new List<ChatMessage>(request.Messages);
        var totalUsage = new TokenUsage();
        var iterations = 0;
        CompletionDiagnostics? finalDiagnostics = null;
        GroundingResult? grounding = null;
        var baseReq = BuildProviderRequest(request, tools: tools);

        while (iterations < maxIterations)
        {
            iterations++;
            var providerReq = baseReq with { Messages = messages };
            var response = await provider.CompleteAsync(providerReq, ct).ConfigureAwait(false);
            totalUsage += response.Usage;
            grounding = GroundingResult.Combine(grounding, response.Grounding);

            if (response.ToolCalls is null or { Count: 0 })
            {
                return new ToolExecutionResponse<string>
                {
                    Content = response.Content,
                    Usage = totalUsage,
                    Iterations = iterations,
                    Messages = messages,
                    Diagnostics = finalDiagnostics ?? BuildDiagnostics(descriptor, response),
                    Grounding = grounding,
                };
            }

            finalDiagnostics = BuildDiagnostics(descriptor, response);

            messages.Add(new ChatMessage
            {
                Role = ChatRole.Assistant,
                Content = response.Content,
                ToolCalls = response.ToolCalls,
            });

            foreach (var toolCall in response.ToolCalls)
            {
                var result = await toolExecutor(toolCall).ConfigureAwait(false);
                messages.Add(new ChatMessage
                {
                    Role = ChatRole.Tool,
                    ToolCallId = result.ToolCallId,
                    Content = result.Result,
                });
            }
        }

        throw new InvalidOperationException($"Tool execution exceeded {maxIterations} iterations");
    }

    public async Task<ToolExecutionResponse<T>> CompleteWithToolsAsync<T>(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        JsonTypeInfo<T> jsonTypeInfo,
        int maxIterations = 20,
        CancellationToken ct = default)
    {
        ValidateGroundingRequest(request);
        var descriptor = AiModelCatalog.Get(request.Model);
        var provider = GetProvider(descriptor.Provider);
        var messages = new List<ChatMessage>(request.Messages);
        var totalUsage = new TokenUsage();
        var iterations = 0;
        CompletionDiagnostics? finalDiagnostics = null;
        GroundingResult? grounding = null;
        var baseReq = BuildProviderRequest(request, tools: tools);

        while (iterations < maxIterations)
        {
            iterations++;
            var providerReq = baseReq with { Messages = messages };
            var response = await provider.CompleteAsync(providerReq, ct).ConfigureAwait(false);
            totalUsage += response.Usage;
            grounding = GroundingResult.Combine(grounding, response.Grounding);

            if (response.ToolCalls is null or { Count: 0 })
            {
                var content = JsonSerializer.Deserialize(response.Content, jsonTypeInfo)!;

                return new ToolExecutionResponse<T>
                {
                    Content = content,
                    Usage = totalUsage,
                    Iterations = iterations,
                    Messages = messages,
                    Diagnostics = finalDiagnostics ?? BuildDiagnostics(descriptor, response),
                    Grounding = grounding,
                };
            }

            finalDiagnostics = BuildDiagnostics(descriptor, response);

            messages.Add(new ChatMessage
            {
                Role = ChatRole.Assistant,
                Content = response.Content,
                ToolCalls = response.ToolCalls,
            });

            foreach (var toolCall in response.ToolCalls)
            {
                var result = await toolExecutor(toolCall).ConfigureAwait(false);
                messages.Add(new ChatMessage
                {
                    Role = ChatRole.Tool,
                    ToolCallId = result.ToolCallId,
                    Content = result.Result,
                });
            }
        }

        throw new InvalidOperationException($"Tool execution exceeded {maxIterations} iterations");
    }

    public async IAsyncEnumerable<ToolStreamEvent> StreamWithToolsAsync(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        int maxIterations = 20,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ValidateGroundingRequest(request);
        var provider = GetProvider(request.Model.GetProvider());
        var messages = new List<ChatMessage>(request.Messages);
        var totalUsage = new TokenUsage();
        var iterations = 0;
        var baseReq = BuildProviderRequest(request, tools: tools);
        GroundingResult? grounding = null;

        while (iterations < maxIterations)
        {
            iterations++;
            var providerReq = baseReq with { Messages = messages };

            var pendingToolCalls = new List<ToolCallRequest>();
            var contentBuilder = new System.Text.StringBuilder();
            TokenUsage? streamUsage = null;

            await foreach (var e in provider.StreamAsync(providerReq, ct).ConfigureAwait(false))
            {
                if (e.TextDelta is not null)
                {
                    contentBuilder.Append(e.TextDelta);
                    yield return new TextDeltaEvent(e.TextDelta);
                }

                if (e.ToolCall is not null)
                {
                    pendingToolCalls.Add(e.ToolCall);
                    yield return new ToolCallDeltaEvent(e.ToolCall.Id, e.ToolCall.Name, e.ToolCall.Arguments);
                }

                if (e.Usage is not null)
                {
                    streamUsage = e.Usage;
                }

                if (e.Grounding is not null)
                {
                    grounding = GroundingResult.Combine(grounding, e.Grounding);
                }
            }

            if (streamUsage is not null)
            {
                totalUsage += streamUsage.Value;
            }

            if (pendingToolCalls.Count == 0)
            {
                yield return new CompletedEvent(totalUsage, grounding);
                yield break;
            }

            messages.Add(new ChatMessage
            {
                Role = ChatRole.Assistant,
                Content = contentBuilder.ToString(),
                ToolCalls = pendingToolCalls,
            });

            foreach (var toolCall in pendingToolCalls)
            {
                var result = await toolExecutor(toolCall).ConfigureAwait(false);
                messages.Add(new ChatMessage
                {
                    Role = ChatRole.Tool,
                    ToolCallId = result.ToolCallId,
                    Content = result.Result,
                });

                yield return new ToolResultAddedEvent(result.ToolCallId, result.Result);
            }
        }

        throw new InvalidOperationException($"Tool execution exceeded {maxIterations} iterations");
    }

    public async Task<EmbeddingResponse> EmbedAsync(EmbeddingRequest request, CancellationToken ct = default)
    {
        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = new ProviderEmbeddingRequest
        {
            ModelApiName = request.Model.ToApiName(),
            Input = request.Input,
            Dimensions = request.Dimensions,
        };

        var response = await provider.EmbedAsync(providerRequest, ct).ConfigureAwait(false);

        return new EmbeddingResponse
        {
            Embedding = response.Embedding,
            Usage = response.Usage,
            Model = request.Model,
        };
    }

    public async Task<BatchEmbeddingResponse> EmbedBatchAsync(BatchEmbeddingRequest request, CancellationToken ct = default)
    {
        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = new ProviderBatchEmbeddingRequest
        {
            ModelApiName = request.Model.ToApiName(),
            Inputs = request.Inputs,
            Dimensions = request.Dimensions,
        };

        var response = await provider.EmbedBatchAsync(providerRequest, ct).ConfigureAwait(false);

        return new BatchEmbeddingResponse
        {
            Embeddings = response.Embeddings,
            Usage = response.Usage,
            Model = request.Model,
        };
    }

    public async Task<byte[]> GenerateImageAsync(ImageGenerationRequest request, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = new ProviderImageRequest
        {
            ModelApiName = request.Model.ToApiName(),
            Prompt = request.Prompt,
            Size = request.Size,
            Quality = request.Quality,
        };

        return await provider.GenerateImageAsync(providerRequest, ct).ConfigureAwait(false);
    }

    private IProvider GetProvider(ProviderType providerType)
    {
        if (_providers.TryGetValue(providerType, out var provider))
        {
            return provider;
        }

        throw new InvalidOperationException(
            $"Provider {providerType} is not configured. Call the appropriate Add method on AiInferenceOptions.");
    }

    private static ProviderRequest BuildProviderRequest(
        CompletionRequest request,
        string? jsonSchemaOverride = null,
        IReadOnlyList<ToolDefinition>? tools = null)
    {
        return new ProviderRequest
        {
            ModelApiName = request.Model.ToApiName(),
            PreferredEndpoint = AiModelCatalog.Get(request.Model).PreferredEndpoint,
            Messages = CloneMessages(request.Messages),
            SystemMessage = request.SystemMessage,
            Temperature = request.Temperature,
            MaxTokens = request.MaxTokens,
            ThinkingBudget = request.ThinkingBudget,
            EnableCache = request.EnableCache,
            JsonSchema = jsonSchemaOverride ?? request.JsonSchema,
            TopP = request.TopP,
            StopSequences = request.StopSequences is null ? null : [.. request.StopSequences],
            Tools = CloneTools(tools),
            EnableWebSearch = request.EnableWebSearch,
            EnableXSearch = request.EnableXSearch,
            Grounding = request.Grounding,
        };
    }

    private static void ValidateGroundingRequest(CompletionRequest request)
    {
        var descriptor = AiModelCatalog.Get(request.Model);
        var grounding = request.Grounding;
        var webSearch = request.EnableWebSearch || grounding?.EnableWebSearch == true;
        var xSearch = request.EnableXSearch || grounding?.EnableXSearch == true;

        if (webSearch && !descriptor.Capabilities.SupportsWebGrounding)
        {
            throw new NotSupportedException($"{descriptor.DisplayName} does not support web grounding.");
        }
        if (grounding?.EnableImageSearch == true && !descriptor.Capabilities.SupportsImageSearch)
        {
            throw new NotSupportedException($"{descriptor.DisplayName} does not support hosted image search.");
        }
        if (xSearch && !descriptor.Capabilities.SupportsXSearch)
        {
            throw new NotSupportedException($"{descriptor.DisplayName} does not support X search.");
        }
        if (grounding is not null && grounding.MaxSearches <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Grounding MaxSearches must be greater than zero.");
        }
        if (grounding?.AllowedDomains is { Count: > 0 } && grounding.BlockedDomains is { Count: > 0 })
        {
            throw new ArgumentException("Grounding cannot specify both allowed and blocked domains.", nameof(request));
        }
    }

    private static IReadOnlyList<ChatMessage> CloneMessages(IReadOnlyList<ChatMessage> messages)
    {
        var clones = new ChatMessage[messages.Count];
        for (var i = 0; i < messages.Count; i++)
        {
            clones[i] = CloneMessage(messages[i]);
        }

        return clones;
    }

    private static ChatMessage CloneMessage(ChatMessage message)
    {
        return new ChatMessage
        {
            Role = message.Role,
            Content = message.Content,
            ToolCallId = message.ToolCallId,
            ToolCalls = CloneToolCalls(message.ToolCalls),
            Attachments = CloneAttachments(message.Attachments),
        };
    }

    private static ToolCallRequest CloneToolCallRequest(ToolCallRequest toolCall)
    {
        return new ToolCallRequest
        {
            Id = toolCall.Id,
            Name = toolCall.Name,
            Arguments = toolCall.Arguments,
            ProviderItemId = toolCall.ProviderItemId,
            ThoughtSignature = toolCall.ThoughtSignature,
        };
    }

    private static DocumentAttachment CloneAttachment(DocumentAttachment attachment)
    {
        return new DocumentAttachment
        {
            FileName = attachment.FileName,
            MimeType = attachment.MimeType,
            Content = attachment.Content.ToArray(),
        };
    }

    private static IReadOnlyList<ToolDefinition>? CloneTools(IReadOnlyList<ToolDefinition>? tools)
    {
        if (tools is null)
        {
            return null;
        }

        var clones = new ToolDefinition[tools.Count];
        for (var i = 0; i < tools.Count; i++)
        {
            var tool = tools[i];
            clones[i] = new ToolDefinition
            {
                Name = tool.Name,
                Description = tool.Description,
                ParametersSchema = tool.ParametersSchema.Clone(),
            };
        }

        return clones;
    }

    private static IReadOnlyList<ToolCallRequest>? CloneToolCalls(IReadOnlyList<ToolCallRequest>? toolCalls)
    {
        if (toolCalls is null)
        {
            return null;
        }

        var clones = new ToolCallRequest[toolCalls.Count];
        for (var i = 0; i < toolCalls.Count; i++)
        {
            clones[i] = CloneToolCallRequest(toolCalls[i]);
        }

        return clones;
    }

    private static IReadOnlyList<DocumentAttachment>? CloneAttachments(IReadOnlyList<DocumentAttachment>? attachments)
    {
        if (attachments is null)
        {
            return null;
        }

        var clones = new DocumentAttachment[attachments.Count];
        for (var i = 0; i < attachments.Count; i++)
        {
            clones[i] = CloneAttachment(attachments[i]);
        }

        return clones;
    }

    private static CompletionDiagnostics BuildDiagnostics(AiModelDescriptor descriptor, ProviderResponse response)
    {
        var stopReason = response.StopReason;
        var returnedContent = !string.IsNullOrWhiteSpace(response.Content);
        var outputMayBeTruncated =
            string.Equals(stopReason, "length", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(stopReason, "max_tokens", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(stopReason, "MAX_TOKENS", StringComparison.OrdinalIgnoreCase) ||
            (!returnedContent && response.Usage.ThinkingTokens > 0);

        return new CompletionDiagnostics
        {
            Provider = descriptor.Provider,
            EndpointFamily = response.EndpointFamily,
            StopReason = stopReason,
            ReturnedContent = returnedContent,
            OutputMayBeTruncated = outputMayBeTruncated,
            Note = response.DiagnosticNote,
        };
    }

    private IProvider CreateProvider(
        ProviderType providerType,
        ProviderCredentials creds,
        HttpClient httpClient,
        Action<InferenceLogLevel, string>? log)
    {
        var context = new ProviderRequestContext
        {
            HttpClient = httpClient,
            BaseUrl = providerType switch
            {
                ProviderType.OpenAi => "https://api.openai.com",
                ProviderType.Anthropic => "https://api.anthropic.com",
                ProviderType.Xai => "https://api.x.ai",
                ProviderType.Google => "https://generativelanguage.googleapis.com",
                _ => throw new ArgumentOutOfRangeException(nameof(providerType))
            },
            ApiKey = creds.ApiKey ?? "",
            Log = log,
            ResiliencePipeline = _resiliencePipeline,
        };

        return providerType switch
        {
            ProviderType.OpenAi => new OpenAiProvider(context),
            ProviderType.Anthropic => new AnthropicProvider(context),
            ProviderType.Google => CreateGoogleProvider(context, creds),
            ProviderType.Xai => new XaiProvider(context),
            _ => throw new ArgumentOutOfRangeException(nameof(providerType))
        };
    }

    private GoogleProvider CreateGoogleProvider(ProviderRequestContext context, ProviderCredentials creds)
    {
        if (creds.ServiceAccountJson is not null)
        {
            var tokenProvider = new GoogleTokenProvider(context.HttpClient, creds.ServiceAccountJson);
            return new GoogleProvider(context, tokenProvider, creds.ProjectId, creds.Location);
        }

        return new GoogleProvider(context);
    }
}
