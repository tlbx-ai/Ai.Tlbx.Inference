namespace Ai.Tlbx.Inference.Resilience;

using Polly;
using Polly.Retry;

internal static class RetryPolicyFactory
{
    public static bool IsRetryableStatusCode(int statusCode)
        => statusCode is 429 or 500 or 502 or 503 or 504;

    public static ResiliencePipeline<HttpResponseMessage> CreateDefault(Action<InferenceLogLevel, string>? log)
    {
        return new ResiliencePipelineBuilder<HttpResponseMessage>()
            .AddRetry(new RetryStrategyOptions<HttpResponseMessage>
            {
                MaxRetryAttempts = 4,
                BackoffType = DelayBackoffType.Exponential,
                Delay = TimeSpan.FromSeconds(1),
                UseJitter = true,
                ShouldHandle = new PredicateBuilder<HttpResponseMessage>()
                    .Handle<RetryableResponseException>()
                    .Handle<HttpRequestException>()
                    .Handle<TaskCanceledException>(ex => !ex.CancellationToken.IsCancellationRequested)
                    .HandleResult(r => IsRetryableStatusCode((int)r.StatusCode)),
                DelayGenerator = args =>
                {
                    if (args.Outcome.Exception is RetryableResponseException retryable && retryable.RetryAfter is { } retryAfterFromException)
                    {
                        return ValueTask.FromResult<TimeSpan?>(retryAfterFromException);
                    }

                    if (args.Outcome.Result?.Headers.RetryAfter?.Delta is { } retryAfter)
                    {
                        return ValueTask.FromResult<TimeSpan?>(retryAfter);
                    }
                    return ValueTask.FromResult<TimeSpan?>(null);
                },
                OnRetry = args =>
                {
                    var statusCode = args.Outcome.Result?.StatusCode ?? args.Outcome.Exception switch
                    {
                        HttpRequestException httpRequestException => httpRequestException.StatusCode,
                        _ => null,
                    };
                    log?.Invoke(InferenceLogLevel.Warning,
                        $"Retry {args.AttemptNumber} after {args.RetryDelay.TotalSeconds:F1}s" +
                        (statusCode.HasValue ? $" for HTTP {(int)statusCode.Value}" : $" for {args.Outcome.Exception?.GetType().Name}"));
                    return ValueTask.CompletedTask;
                }
            })
            .AddTimeout(TimeSpan.FromMinutes(3))
            .Build();
    }

}
