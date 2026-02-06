namespace Ai.Tlbx.Inference;

public abstract record ToolStreamEvent;

public sealed record TextDeltaEvent(string Text) : ToolStreamEvent;

public sealed record ToolCallDeltaEvent(string ToolCallId, string Name, string ArgumentsDelta) : ToolStreamEvent;

public sealed record ToolResultAddedEvent(string ToolCallId, string Result) : ToolStreamEvent;

public sealed record CompletedEvent(TokenUsage Usage) : ToolStreamEvent;
