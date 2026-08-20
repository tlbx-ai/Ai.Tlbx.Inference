namespace Ai.Tlbx.Inference;

public sealed record GroundingResult
{
    public IReadOnlyList<GroundingSource> Sources { get; init; } = [];
    public IReadOnlyList<GroundingImage> Images { get; init; } = [];
    public IReadOnlyList<string> SearchQueries { get; init; } = [];
    public GroundingUsage Usage { get; init; } = new();

    public bool HasResults => Sources.Count > 0 || Images.Count > 0 || SearchQueries.Count > 0;

    public static GroundingResult? Combine(GroundingResult? left, GroundingResult? right)
    {
        if (left is null) return right;
        if (right is null) return left;

        return new GroundingResult
        {
            Sources = left.Sources.Concat(right.Sources)
                .DistinctBy(source => source.Url, StringComparer.OrdinalIgnoreCase)
                .ToArray(),
            Images = left.Images.Concat(right.Images)
                .DistinctBy(image => image.Url, StringComparer.OrdinalIgnoreCase)
                .ToArray(),
            SearchQueries = left.SearchQueries.Concat(right.SearchQueries)
                .Where(query => !string.IsNullOrWhiteSpace(query))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray(),
            Usage = left.Usage + right.Usage
        };
    }
}

public sealed record GroundingSource
{
    public required string Url { get; init; }
    public string? Title { get; init; }
    public string? CitedText { get; init; }
    public int? StartIndex { get; init; }
    public int? EndIndex { get; init; }
}

public sealed record GroundingImage
{
    public required string Url { get; init; }
    public string? SourceUrl { get; init; }
    public string? ThumbnailUrl { get; init; }
    public string? Caption { get; init; }
}

public readonly record struct GroundingUsage
{
    public int WebSearchCalls { get; init; }
    public int ImageSearchCalls { get; init; }
    public int XSearchCalls { get; init; }

    public int TotalSearchCalls => WebSearchCalls + ImageSearchCalls + XSearchCalls;

    public static GroundingUsage operator +(GroundingUsage left, GroundingUsage right) => new()
    {
        WebSearchCalls = left.WebSearchCalls + right.WebSearchCalls,
        ImageSearchCalls = left.ImageSearchCalls + right.ImageSearchCalls,
        XSearchCalls = left.XSearchCalls + right.XSearchCalls
    };
}
