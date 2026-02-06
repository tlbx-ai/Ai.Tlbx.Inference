namespace Ai.Tlbx.Inference;

public enum EmbeddingModel
{
    TextEmbedding3Large,
    TextEmbedding3Small,
    GeminiEmbedding001
}

public static class EmbeddingModelExtensions
{
    public static string ToApiName(this EmbeddingModel model) => model switch
    {
        EmbeddingModel.TextEmbedding3Large => "text-embedding-3-large",
        EmbeddingModel.TextEmbedding3Small => "text-embedding-3-small",
        EmbeddingModel.GeminiEmbedding001 => "gemini-embedding-001",
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static int GetDefaultDimensions(this EmbeddingModel model) => model switch
    {
        EmbeddingModel.TextEmbedding3Large => 3072,
        EmbeddingModel.TextEmbedding3Small => 1536,
        EmbeddingModel.GeminiEmbedding001 => 3072,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    public static ProviderType GetProvider(this EmbeddingModel model) => model switch
    {
        EmbeddingModel.TextEmbedding3Large or EmbeddingModel.TextEmbedding3Small => ProviderType.OpenAi,
        EmbeddingModel.GeminiEmbedding001 => ProviderType.Google,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };
}
