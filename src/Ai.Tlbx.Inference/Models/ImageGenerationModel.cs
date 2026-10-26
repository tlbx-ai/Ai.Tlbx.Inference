namespace Ai.Tlbx.Inference;

public enum ImageGenerationModel
{
    Gemini25FlashImage,
    GptImage2,
    GptImage15,
    GptImage1,
    GptImage1Mini,
    GptImage25Sunburst,
    GptImage25Flare,
    GeminiNanoBanana21,
    Gemini31FlashImage,
    Gemini3ProImage
}

public static class ImageGenerationModelExtensions
{
    public static string ToApiName(this ImageGenerationModel model) => model switch
    {
        ImageGenerationModel.Gemini25FlashImage => "gemini-2.5-flash-image",
        ImageGenerationModel.GptImage2 => "gpt-image-2",
        ImageGenerationModel.GptImage15 => "gpt-image-1.5",
        ImageGenerationModel.GptImage1 => "gpt-image-1",
        ImageGenerationModel.GptImage1Mini => "gpt-image-1-mini",
        ImageGenerationModel.GptImage25Sunburst => "gpt-image-2.5-sunburst",
        ImageGenerationModel.GptImage25Flare => "gpt-image-2.5-flare",
        ImageGenerationModel.GeminiNanoBanana21 => "gemini-nano-banana-2.1",
        ImageGenerationModel.Gemini31FlashImage => "gemini-3.1-flash-image",
        ImageGenerationModel.Gemini3ProImage => "gemini-3-pro-image",
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static ProviderType GetProvider(this ImageGenerationModel model) => model switch
    {
        ImageGenerationModel.Gemini25FlashImage or ImageGenerationModel.GeminiNanoBanana21
            or ImageGenerationModel.Gemini31FlashImage or ImageGenerationModel.Gemini3ProImage => ProviderType.Google,
        ImageGenerationModel.GptImage2
            or ImageGenerationModel.GptImage15
            or ImageGenerationModel.GptImage1
            or ImageGenerationModel.GptImage1Mini or ImageGenerationModel.GptImage25Sunburst
            or ImageGenerationModel.GptImage25Flare => ProviderType.OpenAi,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };
}
