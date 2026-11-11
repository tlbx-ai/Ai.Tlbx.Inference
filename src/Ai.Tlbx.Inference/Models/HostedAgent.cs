namespace Ai.Tlbx.Inference;

/// <summary>Options for a Codex-harness session with an OpenAI-hosted Linux workspace.</summary>
public sealed record HostedAgentOptions
{
    public required string Model { get; init; }
    public string? Instructions { get; init; }
    public string? ReasoningEffort { get; init; }
    public bool EnableWebSearch { get; init; }
    public bool AllowNetworkAccess { get; init; }
    public IReadOnlyList<HostedAgentMcpServer> McpServers { get; init; } = [];
}

/// <summary>An HTTP MCP server reached directly by OpenAI's agent service.</summary>
public sealed record HostedAgentMcpServer
{
    public required string Label { get; init; }
    public required Uri Url { get; init; }
    public IReadOnlyList<string>? AllowedTools { get; init; }
}

public sealed record HostedAgentSession
{
    public required string Id { get; init; }
    public required string Status { get; init; }
    public string? EnvironmentId { get; init; }
    public string? Error { get; init; }
}

/// <summary>One live event; RawJson preserves event fields the beta API may add.</summary>
public sealed record HostedAgentEvent
{
    public required string Type { get; init; }
    public required string RawJson { get; init; }
    public string? SessionId { get; init; }
    public string? TurnId { get; init; }
    public string? ItemId { get; init; }
    public string? Delta { get; init; }
    public string? Text { get; init; }
    public string? Error { get; init; }
    public bool? IsRootTurn { get; init; }
}

public sealed record HostedAgentItem
{
    public required string Id { get; init; }
    public required string Type { get; init; }
    public required string RawJson { get; init; }
    public string? Text { get; init; }
}

public sealed record HostedAgentTurn
{
    public required string Id { get; init; }
    public required string Status { get; init; }
    public string? SubagentId { get; init; }
    public TokenUsage? Usage { get; init; }
}
