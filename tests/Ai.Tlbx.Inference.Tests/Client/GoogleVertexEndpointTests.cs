using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Client;

public sealed class GoogleVertexEndpointTests
{
    [Theory]
    [InlineData("global", "aiplatform.googleapis.com")]
    [InlineData("europe-west4", "europe-west4-aiplatform.googleapis.com")]
    public async Task CompletionAndStreamingUseCorrectVertexHostAndOAuth(string location, string host)
    {
        using var key = RSA.Create(2048);
        var credentials = JsonSerializer.Serialize(new
        {
            client_email = "synthetic@project.iam.gserviceaccount.com", private_key = key.ExportPkcs8PrivateKeyPem()
        });
        var options = new AiInferenceOptions().AddGoogle(credentials, "synthetic-project", location);
        var modelRequests = new List<Uri>();
        var tokenRequests = 0;
        const string response = "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Source text\"}]},\"finishReason\":\"STOP\"}]}";
        using var http = new HttpClient(new MockHttpHandler(request =>
        {
            if (request.RequestUri!.Host == "oauth2.googleapis.com")
            {
                tokenRequests++;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("{\"access_token\":\"synthetic-token\",\"expires_in\":3600}")
                });
            }
            modelRequests.Add(request.RequestUri);
            Assert.Equal(host, request.RequestUri.Host);
            Assert.Contains($"/projects/synthetic-project/locations/{location}/publishers/google/models/", request.RequestUri.AbsolutePath);
            Assert.Equal("Bearer", request.Headers.Authorization!.Scheme);
            Assert.Equal("synthetic-token", request.Headers.Authorization.Parameter);
            Assert.DoesNotContain("key=", request.RequestUri.Query);
            var streaming = request.RequestUri.AbsolutePath.EndsWith(":streamGenerateContent", StringComparison.Ordinal);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = streaming ? new StringContent("data: " + response + "\n\n", Encoding.UTF8, "text/event-stream") : new StringContent(response)
            });
        }));
        var client = new AiInferenceClient(http, options);
        var completion = await client.CompleteAsync(Request());
        Assert.Equal("Source text", completion.Content);
        var chunks = new List<string>();
        await foreach (var chunk in client.StreamAsync(Request())) chunks.Add(chunk);
        Assert.NotEmpty(chunks);
        Assert.Equal(2, modelRequests.Count);
        Assert.Equal(1, tokenRequests);
    }

    private static CompletionRequest Request() => new()
    {
        Model = AiModel.Gemini3FlashPreview,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Synthetic input" }]
    };
}
