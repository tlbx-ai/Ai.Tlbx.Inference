namespace Ai.Tlbx.Inference;

public sealed record ImageEditRequest
{
    public required string Prompt { get; init; }
    public required IReadOnlyList<ImageEditInput> Images { get; init; }
    public ImageGenerationModel Model { get; init; } = ImageGenerationModel.GptImage2;
    public string? Size { get; init; }
    public string? Quality { get; init; }
}

public sealed record ImageEditInput
{
    public required ReadOnlyMemory<byte> Content { get; init; }
    public required string FileName { get; init; }
    public required string MimeType { get; init; }
}
