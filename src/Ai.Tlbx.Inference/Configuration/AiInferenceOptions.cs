using Polly;

namespace Ai.Tlbx.Inference.Configuration;

public sealed class AiInferenceOptions
{
    internal Dictionary<ProviderType, ProviderCredentials> Providers { get; } = new();
    internal Action<InferenceLogLevel, string>? LogAction { get; private set; }
    internal ResiliencePipeline<HttpResponseMessage>? CustomRetryPolicy { get; private set; }

    public AiInferenceOptions AddOpenAi(string apiKey)
    {
        Providers[ProviderType.OpenAi] = new() { ApiKey = apiKey };
        return this;
    }

    public AiInferenceOptions AddAnthropic(string apiKey)
    {
        Providers[ProviderType.Anthropic] = new() { ApiKey = apiKey };
        return this;
    }

    public AiInferenceOptions AddGoogle(string apiKey)
    {
        Providers[ProviderType.Google] = new() { ApiKey = apiKey };
        return this;
    }

    public AiInferenceOptions AddGoogle(string serviceAccountJson, string projectId, string location)
    {
        Providers[ProviderType.Google] = new()
        {
            ServiceAccountJson = serviceAccountJson,
            ProjectId = projectId,
            Location = location
        };
        return this;
    }

    public AiInferenceOptions AddXai(string apiKey)
    {
        Providers[ProviderType.Xai] = new() { ApiKey = apiKey };
        return this;
    }

    public AiInferenceOptions WithLogging(Action<InferenceLogLevel, string> logAction)
    {
        LogAction = logAction;
        return this;
    }

    public AiInferenceOptions WithRetryPolicy(ResiliencePipeline<HttpResponseMessage> policy)
    {
        CustomRetryPolicy = policy;
        return this;
    }
}
