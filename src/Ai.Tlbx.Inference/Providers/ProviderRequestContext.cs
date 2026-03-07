using Polly;
using Ai.Tlbx.Inference.Resilience;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class ProviderRequestContext
{
    public required HttpClient HttpClient { get; init; }
    public required string BaseUrl { get; init; }
    public required string ApiKey { get; init; }
    public ResiliencePipeline<HttpResponseMessage> ResiliencePipeline { get; init; } = RetryPolicyFactory.CreateDefault(null);
    public Action<InferenceLogLevel, string>? Log { get; init; }

    public Task<HttpResponseMessage> SendAsync(
        Func<HttpRequestMessage> requestFactory,
        CancellationToken ct)
        => SendAsync(requestFactory, HttpCompletionOption.ResponseContentRead, ct);

    public Task<HttpResponseMessage> SendAsync(
        Func<HttpRequestMessage> requestFactory,
        HttpCompletionOption completionOption,
        CancellationToken ct)
        => SendAsync(_ => Task.FromResult(requestFactory()), completionOption, ct);

    public Task<HttpResponseMessage> SendAsync(
        Func<CancellationToken, Task<HttpRequestMessage>> requestFactory,
        CancellationToken ct)
        => SendAsync(requestFactory, HttpCompletionOption.ResponseContentRead, ct);

    public Task<HttpResponseMessage> SendAsync(
        Func<CancellationToken, Task<HttpRequestMessage>> requestFactory,
        HttpCompletionOption completionOption,
        CancellationToken ct)
    {
        return ResiliencePipeline.ExecuteAsync(async token =>
        {
            using var request = await requestFactory(token).ConfigureAwait(false);
            var response = await HttpClient.SendAsync(request, completionOption, token).ConfigureAwait(false);

            if (RetryPolicyFactory.IsRetryableStatusCode((int)response.StatusCode))
            {
                var statusCode = response.StatusCode;
                var retryAfter = response.Headers.RetryAfter?.Delta;
                response.Dispose();
                throw new RetryableResponseException(statusCode, retryAfter);
            }

            return response;
        }, ct).AsTask();
    }
}
