namespace Ai.Tlbx.Inference;

public static class AiModelValidator
{
    public static AiModelValidationResult ValidateForCompletion(
        AiModel model,
        bool streaming = false,
        bool documentAttachments = false,
        bool tools = false,
        bool structuredOutput = false,
        bool webSearch = false,
        bool xSearch = false)
    {
        var descriptor = AiModelCatalog.Get(model);
        var errors = new List<string>();
        var warnings = new List<string>();

        if (streaming && !descriptor.Capabilities.SupportsStreaming)
        {
            errors.Add("Streaming is not supported by this model.");
        }

        if (documentAttachments && !descriptor.Capabilities.SupportsDocumentAttachments)
        {
            errors.Add("Document attachments are not supported by this model.");
        }

        if (tools && !descriptor.Capabilities.SupportsTools)
        {
            errors.Add("Tool calling is not supported by this model.");
        }

        if (structuredOutput && !descriptor.Capabilities.SupportsStructuredOutput)
        {
            errors.Add("Structured output is not supported by this model.");
        }

        if (webSearch && !descriptor.Capabilities.SupportsWebGrounding)
        {
            errors.Add("Web grounding is not supported by this model.");
        }

        if (xSearch && !descriptor.Capabilities.SupportsXSearch)
        {
            errors.Add("X search is not supported by this model.");
        }

        if (descriptor.Capabilities.IsPreview)
        {
            warnings.Add("This model is a preview model and may change behavior or identifiers without notice.");
        }

        if (descriptor.Capabilities.RequiresReasoningBudgetHeadroom)
        {
            warnings.Add("This model may consume output budget for reasoning before returning visible text.");
        }

        if (descriptor.PreferredEndpoint == ModelEndpointFamily.Responses)
        {
            warnings.Add("This model works best through the Responses API path.");
        }

        if (!string.IsNullOrWhiteSpace(descriptor.Notes))
        {
            warnings.Add(descriptor.Notes!);
        }

        return new AiModelValidationResult
        {
            Model = model,
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings,
        };
    }
}
