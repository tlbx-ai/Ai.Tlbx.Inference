namespace Ai.Tlbx.Inference;

public sealed record ToolCallResult
{
    public required string ToolCallId { get; init; }
    public required string Result { get; init; }
    public bool IsError { get; init; }
}
