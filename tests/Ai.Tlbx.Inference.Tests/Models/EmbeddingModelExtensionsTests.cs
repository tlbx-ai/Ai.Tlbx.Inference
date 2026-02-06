namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class EmbeddingModelExtensionsTests
{
    [Theory]
    [InlineData(EmbeddingModel.TextEmbedding3Large, "text-embedding-3-large")]
    [InlineData(EmbeddingModel.TextEmbedding3Small, "text-embedding-3-small")]
    [InlineData(EmbeddingModel.GeminiEmbedding001, "gemini-embedding-001")]
    public void ToApiName_ReturnsExpectedValue(EmbeddingModel model, string expected)
    {
        Assert.Equal(expected, model.ToApiName());
    }

    [Theory]
    [InlineData(EmbeddingModel.TextEmbedding3Large, ProviderType.OpenAi)]
    [InlineData(EmbeddingModel.TextEmbedding3Small, ProviderType.OpenAi)]
    [InlineData(EmbeddingModel.GeminiEmbedding001, ProviderType.Google)]
    public void GetProvider_ReturnsCorrectType(EmbeddingModel model, ProviderType expected)
    {
        Assert.Equal(expected, model.GetProvider());
    }

    [Theory]
    [InlineData(EmbeddingModel.TextEmbedding3Large, 3072)]
    [InlineData(EmbeddingModel.TextEmbedding3Small, 1536)]
    [InlineData(EmbeddingModel.GeminiEmbedding001, 3072)]
    public void GetDefaultDimensions_ReturnsExpectedValue(EmbeddingModel model, int expected)
    {
        Assert.Equal(expected, model.GetDefaultDimensions());
    }
}
