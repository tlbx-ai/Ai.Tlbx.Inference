namespace Ai.Tlbx.Inference;

/// <summary>Optional accounting boundary around each provider call in a tool loop.</summary>
public interface IToolIterationObserver
{
    /// <summary>
    /// Runs before the provider call. Throwing prevents that call, for example when a budget
    /// reservation is denied. No finished notification is sent when this method throws.
    /// </summary>
    ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken);

    /// <summary>
    /// Runs once after a started call, before any tools execute, including on failure or stream
    /// disposal. It is deliberately not passed the canceled request token. Implementations must
    /// bound their own persistence work. A callback failure stops the loop and is propagated.
    /// </summary>
    ValueTask OnFinishedAsync(ToolIterationResult result);
}

public sealed record ToolIterationRequest
{
    public required int Iteration { get; init; }
    /// <summary>The current conversation, including earlier tool results; the observer is cleared.</summary>
    public required CompletionRequest Request { get; init; }
    public required IReadOnlyList<ToolDefinition> Tools { get; init; }
}

public sealed record ToolIterationResult
{
    public required int Iteration { get; init; }
    /// <summary>Latest reported usage for this call only. Null means unknown, not zero.</summary>
    public TokenUsage? Usage { get; init; }
    public GroundingResult? Grounding { get; init; }
    /// <summary>
    /// Whether the provider response finished. False means failure, cancellation or early stream
    /// disposal; reported usage may then be partial. Earlier calls have already been notified.
    /// This does not report the outcome of subsequent tool execution or result deserialization.
    /// </summary>
    public required bool ResponseCompleted { get; init; }
}
