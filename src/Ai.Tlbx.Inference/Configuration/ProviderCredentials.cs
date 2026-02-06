namespace Ai.Tlbx.Inference.Configuration;

internal sealed record ProviderCredentials
{
    public string? ApiKey { get; init; }
    public string? ServiceAccountJson { get; init; }
    public string? ProjectId { get; init; }
    public string? Location { get; init; }
}
