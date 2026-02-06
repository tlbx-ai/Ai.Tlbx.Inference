namespace Ai.Tlbx.Inference.Resilience;

using Polly;
using Polly.Retry;

internal static class RetryPolicyFactory
{
    private static readonly HashSet<int> _retryableStatusCodes = [429, 500, 502, 503, 504];

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
                    .Handle<HttpRequestException>()
                    .Handle<TaskCanceledException>(ex => !ex.CancellationToken.IsCancellationRequested)
                    .HandleResult(r => _retryableStatusCodes.Contains((int)r.StatusCode)),
                DelayGenerator = args =>
                {
                    if (args.Outcome.Result?.Headers.RetryAfter?.Delta is { } retryAfter)
                    {
                        return ValueTask.FromResult<TimeSpan?>(retryAfter);
                    }
                    return ValueTask.FromResult<TimeSpan?>(null);
                },
                OnRetry = args =>
                {
                    var statusCode = args.Outcome.Result?.StatusCode;
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
