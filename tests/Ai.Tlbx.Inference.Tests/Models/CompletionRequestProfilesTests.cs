namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class CompletionRequestProfilesTests
{
    [Fact]
    public void CreateSmoke_UsesCatalogDefaults()
    {
        var descriptor = AiModelCatalog.Get(AiModel.Gemini31FlashLitePreview);
        var request = CompletionRequestProfiles.CreateSmoke(AiModel.Gemini31FlashLitePreview);

        Assert.Equal(AiModel.Gemini31FlashLitePreview, request.Model);
        Assert.Equal(descriptor.Capabilities.DefaultSmokeMaxTokens, request.MaxTokens);
        Assert.Equal("Reply with exactly: hi", request.Messages[0].Content);
    }

    [Fact]
    public void CreateSmokeRetry_UsesRetryBudget()
    {
        var descriptor = AiModelCatalog.Get(AiModel.Gpt52Pro);
        var request = CompletionRequestProfiles.CreateSmokeRetry(AiModel.Gpt52Pro);

        Assert.Equal(descriptor.Capabilities.RetrySmokeMaxTokens, request.MaxTokens);
    }

    [Fact]
    public void CreateStreamingSmoke_UsesStreamingBudget()
    {
        var descriptor = AiModelCatalog.Get(AiModel.Gemini31FlashLitePreview);
        var request = CompletionRequestProfiles.CreateStreamingSmoke(AiModel.Gemini31FlashLitePreview);

        Assert.Equal(descriptor.Capabilities.StreamingSmokeMaxTokens, request.MaxTokens);
    }

    [Fact]
    public void CreateAttachmentSmoke_UsesAttachmentBudget()
    {
        var descriptor = AiModelCatalog.Get(AiModel.Gpt54);
        var request = CompletionRequestProfiles.CreateAttachmentSmoke(
            AiModel.Gpt54,
            new DocumentAttachment
            {
                FileName = "test.pdf",
                MimeType = "application/pdf",
                Content = "pdf"u8.ToArray(),
            });

        Assert.Equal(descriptor.Capabilities.AttachmentSmokeMaxTokens, request.MaxTokens);
    }
}
