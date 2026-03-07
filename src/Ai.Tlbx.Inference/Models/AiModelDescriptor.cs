namespace Ai.Tlbx.Inference;

public sealed record AiModelDescriptor
{
    public required AiModel Model { get; init; }
    public required string ApiName { get; init; }
    public required string DisplayName { get; init; }
    public required ProviderType Provider { get; init; }
    public required int ContextWindow { get; init; }
    public required ModelEndpointFamily PreferredEndpoint { get; init; }
    public required AiModelCapabilities Capabilities { get; init; }
    public string? Notes { get; init; }
}
