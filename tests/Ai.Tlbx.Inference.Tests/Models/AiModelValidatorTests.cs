namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class AiModelValidatorTests
{
    [Fact]
    public void ValidateForCompletion_WarnsWhenModelUsesResponsesApi()
    {
        var result = AiModelValidator.ValidateForCompletion(AiModel.Gpt52Pro);

        Assert.True(result.IsValid);
        Assert.Contains(result.Warnings, warning => warning.Contains("Responses API", StringComparison.Ordinal));
    }

    [Fact]
    public void ValidateForCompletion_AllowsProviderNativeWebGrounding()
    {
        var result = AiModelValidator.ValidateForCompletion(AiModel.Grok4, webSearch: true);

        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

}
