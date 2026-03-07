using System.Net;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Client;

public sealed class CompletionDiagnosticsTests
{
    [Fact]
    public async Task CompleteAsync_PopulatesDiagnostics()
    {
        var json = """
        {
            "status": "incomplete",
            "output": [{
                "type": "message",
                "content": []
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "output_tokens_details": { "reasoning_tokens": 20 }
            }
        }
        """;

        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var httpClient = new HttpClient(handler);
        var options = new AiInferenceOptions().AddOpenAi("key");
        var client = new AiInferenceClient(httpClient, options);

        var response = await client.CompleteAsync(new CompletionRequest
        {
            Model = AiModel.Gpt52Pro,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "hi" }],
        });

        Assert.NotNull(response.Diagnostics);
        Assert.Equal(ProviderType.OpenAi, response.Diagnostics!.Provider);
        Assert.Equal(ModelEndpointFamily.Responses, response.Diagnostics.EndpointFamily);
        Assert.True(response.Diagnostics.OutputMayBeTruncated);
        Assert.False(response.Diagnostics.ReturnedContent);
        Assert.NotNull(response.Diagnostics.Note);
    }
}
