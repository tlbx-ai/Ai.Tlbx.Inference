namespace Ai.Tlbx.Inference;

public sealed record ToolCallRequest
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required string Arguments { get; init; }
}
