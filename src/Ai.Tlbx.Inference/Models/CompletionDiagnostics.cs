namespace Ai.Tlbx.Inference;

public sealed record CompletionDiagnostics
{
    public required ProviderType Provider { get; init; }
    public required ModelEndpointFamily EndpointFamily { get; init; }
    public string? StopReason { get; init; }
    public bool ReturnedContent { get; init; }
    public bool OutputMayBeTruncated { get; init; }
    public string? Note { get; init; }
}
