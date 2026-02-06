namespace Ai.Tlbx.Inference;

public enum ChatRole
{
    System,
    User,
    Assistant,
    Tool
}

public sealed record ChatMessage
{
    public required ChatRole Role { get; init; }
    public string? Content { get; init; }
    public string? ToolCallId { get; init; }
    public IReadOnlyList<ToolCallRequest>? ToolCalls { get; init; }
    public IReadOnlyList<DocumentAttachment>? Attachments { get; init; }
}
