namespace Ai.Tlbx.Inference;

public abstract record ToolStreamEvent;

public sealed record TextDeltaEvent(string Text) : ToolStreamEvent;

public sealed record ToolCallDeltaEvent(string ToolCallId, string Name, string ArgumentsDelta) : ToolStreamEvent;

public sealed record ToolResultAddedEvent(string ToolCallId, string Result) : ToolStreamEvent;

public sealed record CompletedEvent(TokenUsage Usage, GroundingResult? Grounding = null, CompactionState? CompactionState = null) : ToolStreamEvent;

/// <summary>Canonical assistant/tool message; assistant calls are committed before execution.</summary>
public sealed record ConversationMessageAddedEvent(ChatMessage Message) : ToolStreamEvent;
public sealed record ConversationCompactedEvent(ConversationCompaction Compaction) : ToolStreamEvent;
