namespace Ai.Tlbx.Inference;

public sealed record GroundingOptions
{
    public bool EnableWebSearch { get; init; } = true;
    public bool EnableImageSearch { get; init; }
    public bool EnableXSearch { get; init; }
    public int MaxSearches { get; init; } = 3;
    public IReadOnlyList<string>? AllowedDomains { get; init; }
    public IReadOnlyList<string>? BlockedDomains { get; init; }
    public GroundingLocation? UserLocation { get; init; }
}

public sealed record GroundingLocation
{
    public string? City { get; init; }
    public string? Region { get; init; }
    public string? Country { get; init; }
    public string? Timezone { get; init; }
}
