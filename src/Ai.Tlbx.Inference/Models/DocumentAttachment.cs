namespace Ai.Tlbx.Inference;

public sealed record DocumentAttachment
{
    public required string FileName { get; init; }
    public required string MimeType { get; init; }
    public required ReadOnlyMemory<byte> Content { get; init; }
}
