namespace Ai.Tlbx.Inference.Providers;

internal sealed class XaiProvider : OpenAiCompatibleProvider
{
    public XaiProvider(ProviderRequestContext context) : base(context)
    {
    }

    protected override string MapReasoningEffort(int thinkingBudget) => thinkingBudget switch
    {
        < 10000 => "low",
        _ => "high",
    };

    public override Task<ProviderEmbeddingResponse> EmbedAsync(
        ProviderEmbeddingRequest request,
        CancellationToken ct)
        => throw new NotSupportedException("xAI does not support embeddings.");

    public override Task<ProviderBatchEmbeddingResponse> EmbedBatchAsync(
        ProviderBatchEmbeddingRequest request,
        CancellationToken ct)
        => throw new NotSupportedException("xAI does not support embeddings.");

    public override Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
        => throw new NotSupportedException("xAI does not support image generation.");
}
