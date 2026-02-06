using System.Runtime.CompilerServices;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Resilience;
using Ai.Tlbx.Inference.Schema;
using Polly;

namespace Ai.Tlbx.Inference;

public sealed class AiInferenceClient : IAiInferenceClient
{
    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

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
        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = BuildProviderRequest(request);
        var response = await provider.CompleteAsync(providerRequest, ct).ConfigureAwait(false);

        return new CompletionResponse<string>
        {
            Content = response.Content,
            Usage = response.Usage,
            Model = request.Model,
            StopReason = response.StopReason,
        };
    }

    public async Task<CompletionResponse<T>> CompleteAsync<T>(CompletionRequest request, CancellationToken ct = default)
    {
        if (typeof(T) == typeof(string))
        {
            var stringResult = await CompleteAsync(request, ct).ConfigureAwait(false);
            return (CompletionResponse<T>)(object)stringResult;
        }

        var schema = request.JsonSchema ?? JsonSchemaGenerator.Generate(typeof(T)).GetRawText();
        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = BuildProviderRequest(request, schema);
        var response = await provider.CompleteAsync(providerRequest, ct).ConfigureAwait(false);

        var content = JsonSerializer.Deserialize<T>(response.Content, _jsonOptions)!;

        return new CompletionResponse<T>
        {
            Content = content,
            Usage = response.Usage,
            Model = request.Model,
            StopReason = response.StopReason,
        };
    }

    public async IAsyncEnumerable<string> StreamAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var provider = GetProvider(request.Model.GetProvider());
        var providerRequest = BuildProviderRequest(request);

        await foreach (var e in provider.StreamAsync(providerRequest, ct).ConfigureAwait(false))
        {
            if (e.TextDelta is not null)
            {
                yield return e.TextDelta;
            }
        }
    }

    public async Task<ToolExecutionResponse<T>> CompleteWithToolsAsync<T>(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        int maxIterations = 20,
        CancellationToken ct = default)
    {
        var provider = GetProvider(request.Model.GetProvider());
        var messages = new List<ChatMessage>(request.Messages);
        var totalUsage = new TokenUsage();
        var iterations = 0;

        while (iterations < maxIterations)
        {
            iterations++;
            var providerReq = BuildProviderRequest(request with { Messages = messages }, tools: tools);
            var response = await provider.CompleteAsync(providerReq, ct).ConfigureAwait(false);
            totalUsage += response.Usage;

            if (response.ToolCalls is null or { Count: 0 })
            {
                var content = typeof(T) == typeof(string)
                    ? (T)(object)response.Content
                    : JsonSerializer.Deserialize<T>(response.Content, _jsonOptions)!;

                return new ToolExecutionResponse<T>
                {
                    Content = content,
                    Usage = totalUsage,
                    Iterations = iterations,
                    Messages = messages,
                };
            }

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
        var provider = GetProvider(request.Model.GetProvider());
        var messages = new List<ChatMessage>(request.Messages);
        var totalUsage = new TokenUsage();
        var iterations = 0;

        while (iterations < maxIterations)
        {
            iterations++;
            var providerReq = BuildProviderRequest(request with { Messages = messages }, tools: tools);

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
            }

            if (streamUsage is not null)
            {
                totalUsage += streamUsage;
            }

            if (pendingToolCalls.Count == 0)
            {
                yield return new CompletedEvent(totalUsage);
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
        var provider = GetProvider(ProviderType.Google);
        var providerRequest = new ProviderImageRequest
        {
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
            Messages = request.Messages,
            SystemMessage = request.SystemMessage,
            Temperature = request.Temperature,
            MaxTokens = request.MaxTokens,
            ThinkingBudget = request.ThinkingBudget,
            EnableCache = request.EnableCache,
            JsonSchema = jsonSchemaOverride ?? request.JsonSchema,
            TopP = request.TopP,
            StopSequences = request.StopSequences,
            Tools = tools,
        };
    }

    private static IProvider CreateProvider(
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

    private static GoogleProvider CreateGoogleProvider(ProviderRequestContext context, ProviderCredentials creds)
    {
        if (creds.ServiceAccountJson is not null)
        {
            var tokenProvider = new GoogleTokenProvider(creds.ServiceAccountJson);
            return new GoogleProvider(context, tokenProvider, creds.ProjectId, creds.Location);
        }

        return new GoogleProvider(context);
    }
}
