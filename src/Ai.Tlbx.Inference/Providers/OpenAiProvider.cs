namespace Ai.Tlbx.Inference.Providers;

internal sealed class OpenAiProvider : OpenAiCompatibleProvider
{
    public OpenAiProvider(ProviderRequestContext context) : base(context)
    {
    }

    protected override string MapReasoningEffort(int thinkingBudget) => thinkingBudget switch
    {
        < 5000 => "low",
        <= 20000 => "medium",
        _ => "high",
    };

    public override Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
        => throw new NotSupportedException("OpenAI image generation is not supported in this library.");
}
