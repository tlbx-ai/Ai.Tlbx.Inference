using System.Text.Json.Serialization;

namespace Ai.Tlbx.Inference.Json;

internal sealed record OpenAiFileUploadResponseDto
{
    public string? Id { get; init; }
}

internal sealed record OpenAiEmbeddingResponseDto
{
    public required OpenAiEmbeddingItemDto[] Data { get; init; }
    public required OpenAiPromptUsageDto Usage { get; init; }
}

internal sealed record OpenAiEmbeddingItemDto
{
    public required float[] Embedding { get; init; }
}

internal sealed record OpenAiPromptUsageDto
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }
}

internal sealed record GoogleEmbeddingResponseDto
{
    public required GoogleEmbeddingDto Embedding { get; init; }
}

internal sealed record GoogleEmbeddingDto
{
    public required float[] Values { get; init; }
}

internal sealed record GoogleBatchEmbeddingResponseDto
{
    public required GoogleEmbeddingDto[] Embeddings { get; init; }
}

internal sealed record GoogleImageGenerationResponseDto
{
    public required GoogleImageCandidateDto[] Candidates { get; init; }
}

internal sealed record GoogleImageCandidateDto
{
    public required GoogleImageContentDto Content { get; init; }
}

internal sealed record GoogleImageContentDto
{
    public required GoogleInlinePartDto[] Parts { get; init; }
}

internal sealed record GoogleInlinePartDto
{
    [JsonPropertyName("inline_data")]
    public GoogleInlineDataDto? InlineData { get; init; }
}

internal sealed record GoogleInlineDataDto
{
    public string? Data { get; init; }
}
