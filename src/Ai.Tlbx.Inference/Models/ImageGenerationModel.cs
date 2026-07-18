namespace Ai.Tlbx.Inference;

public enum ImageGenerationModel
{
    Gemini25FlashImage,
    GptImage2,
    GptImage15,
    GptImage1,
    GptImage1Mini
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
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static ProviderType GetProvider(this ImageGenerationModel model) => model switch
    {
        ImageGenerationModel.Gemini25FlashImage => ProviderType.Google,
        ImageGenerationModel.GptImage2
            or ImageGenerationModel.GptImage15
            or ImageGenerationModel.GptImage1
            or ImageGenerationModel.GptImage1Mini => ProviderType.OpenAi,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };
}
