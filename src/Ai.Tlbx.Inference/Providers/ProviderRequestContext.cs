namespace Ai.Tlbx.Inference.Providers;

internal sealed class ProviderRequestContext
{
    public required HttpClient HttpClient { get; init; }
    public required string BaseUrl { get; init; }
    public required string ApiKey { get; init; }
    public Action<InferenceLogLevel, string>? Log { get; init; }
}
