namespace Ai.Tlbx.Inference;

public static class CompletionRequestProfiles
{
    public static CompletionRequest CreateSmoke(AiModel model, string prompt = "Reply with exactly: hi")
    {
        var descriptor = AiModelCatalog.Get(model);

        return new CompletionRequest
        {
            Model = model,
            MaxTokens = descriptor.Capabilities.DefaultSmokeMaxTokens,
            Messages =
            [
                new ChatMessage
                {
                    Role = ChatRole.User,
                    Content = prompt,
                }
            ],
        };
    }

    public static CompletionRequest CreateSmokeRetry(AiModel model, string prompt = "Reply with exactly: hi")
    {
        var descriptor = AiModelCatalog.Get(model);

        return new CompletionRequest
        {
            Model = model,
            MaxTokens = descriptor.Capabilities.RetrySmokeMaxTokens,
            Messages =
            [
                new ChatMessage
                {
                    Role = ChatRole.User,
                    Content = prompt,
                }
            ],
        };
    }

    public static CompletionRequest CreateStreamingSmoke(
        AiModel model,
        string prompt = "Write exactly eight lines. Start each line with 'Line N:' and make the text long enough to stream in multiple chunks.")
    {
        var descriptor = AiModelCatalog.Get(model);

        return new CompletionRequest
        {
            Model = model,
            MaxTokens = descriptor.Capabilities.StreamingSmokeMaxTokens,
            Messages =
            [
                new ChatMessage
                {
                    Role = ChatRole.User,
                    Content = prompt,
                }
            ],
        };
    }

    public static CompletionRequest CreateAttachmentSmoke(
        AiModel model,
        DocumentAttachment attachment,
        string prompt = "Read the attached document and reply exactly with: Status=GREEN; Code=42")
    {
        var descriptor = AiModelCatalog.Get(model);

        return new CompletionRequest
        {
            Model = model,
            MaxTokens = descriptor.Capabilities.AttachmentSmokeMaxTokens,
            Messages =
            [
                new ChatMessage
                {
                    Role = ChatRole.User,
                    Content = prompt,
                    Attachments = [attachment],
                }
            ],
        };
    }
}
