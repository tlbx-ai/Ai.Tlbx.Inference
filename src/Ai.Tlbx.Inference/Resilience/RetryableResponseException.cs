using System.Net;

namespace Ai.Tlbx.Inference.Resilience;

internal sealed class RetryableResponseException : HttpRequestException
{
    public RetryableResponseException(HttpStatusCode statusCode, TimeSpan? retryAfter)
        : base($"Transient HTTP {(int)statusCode}", null, statusCode)
    {
        RetryAfter = retryAfter;
    }

    public TimeSpan? RetryAfter { get; }
}
