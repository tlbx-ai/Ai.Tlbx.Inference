namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class ImageGenerationModelExtensionsTests
{
    [Theory]
    [InlineData(ImageGenerationModel.Gemini25FlashImage, "gemini-2.5-flash-image")]
    [InlineData(ImageGenerationModel.GptImage2, "gpt-image-2")]
    [InlineData(ImageGenerationModel.GptImage15, "gpt-image-1.5")]
    [InlineData(ImageGenerationModel.GptImage1, "gpt-image-1")]
    [InlineData(ImageGenerationModel.GptImage1Mini, "gpt-image-1-mini")]
    public void ToApiName_ReturnsExpectedValue(ImageGenerationModel model, string expected)
    {
        Assert.Equal(expected, model.ToApiName());
    }

    [Theory]
    [InlineData(ImageGenerationModel.Gemini25FlashImage, ProviderType.Google)]
    [InlineData(ImageGenerationModel.GptImage2, ProviderType.OpenAi)]
    [InlineData(ImageGenerationModel.GptImage15, ProviderType.OpenAi)]
    [InlineData(ImageGenerationModel.GptImage1, ProviderType.OpenAi)]
    [InlineData(ImageGenerationModel.GptImage1Mini, ProviderType.OpenAi)]
    public void GetProvider_ReturnsExpectedValue(ImageGenerationModel model, ProviderType expected)
    {
        Assert.Equal(expected, model.GetProvider());
    }
}
