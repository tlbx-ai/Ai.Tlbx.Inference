namespace Ai.Tlbx.Inference;

public enum ChatRole
{
    System,
    User,
    Assistant,
    Tool,
    /// <summary>Opaque native provider input item (for example an entire compacted window item).</summary>
    Provider
}

public sealed record ChatMessage
{
    public required ChatRole Role { get; init; }
    public string? Content { get; init; }
    /// <summary>Opaque provider continuation data. Preserve unchanged when retaining assistant history.</summary>
    public string? ProviderContentJson { get; init; }
    public string? ToolCallId { get; init; }
    public IReadOnlyList<ToolCallRequest>? ToolCalls { get; init; }
    public IReadOnlyList<DocumentAttachment>? Attachments { get; init; }
}
