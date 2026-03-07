using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class XaiProviderTests
{
    [Fact]
    public async Task CompleteAsync_SendsToXaiBaseUrl()
    {
        HttpRequestMessage? captured = null;
        var json = BuildChatResponse("Hello from Grok!");
        var handler = new MockHttpHandler(async req =>
        {
            captured = req;
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var httpClient = new HttpClient(handler);
        var context = new ProviderRequestContext
        {
            HttpClient = httpClient,
            BaseUrl = "https://api.x.ai",
            ApiKey = "xai-test-key",
        };

        var provider = new XaiProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(captured);
        Assert.StartsWith("https://api.x.ai/v1/chat/completions", captured!.RequestUri?.ToString() ?? "");
    }

    [Fact]
    public async Task CompleteAsync_ParsesContent()
    {
        var json = BuildChatResponse("Grok says hi");
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal("Grok says hi", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_ThinkingBudget_LowThreshold()
    {
        string? capturedBody = null;
        var json = BuildChatResponse("Result");
        var handler = new MockHttpHandler(async req =>
        {
            capturedBody = await req.Content!.ReadAsStringAsync();
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var httpClient = new HttpClient(handler);
        var context = new ProviderRequestContext
        {
            HttpClient = httpClient,
            BaseUrl = "https://api.x.ai",
            ApiKey = "test",
        };

        var provider = new XaiProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "grok-4",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Think" }],
            ThinkingBudget = 5000,
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        Assert.Equal("low", doc.RootElement.GetProperty("reasoning_effort").GetString());
    }

    [Fact]
    public async Task CompleteAsync_ThinkingBudget_HighThreshold()
    {
        string? capturedBody = null;
        var json = BuildChatResponse("Result");
        var handler = new MockHttpHandler(async req =>
        {
            capturedBody = await req.Content!.ReadAsStringAsync();
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var httpClient = new HttpClient(handler);
        var context = new ProviderRequestContext
        {
            HttpClient = httpClient,
            BaseUrl = "https://api.x.ai",
            ApiKey = "test",
        };

        var provider = new XaiProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "grok-4",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Think hard" }],
            ThinkingBudget = 20000,
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        Assert.Equal("high", doc.RootElement.GetProperty("reasoning_effort").GetString());
    }

    [Fact]
    public async Task EmbedAsync_ThrowsNotSupported()
    {
        var provider = CreateProvider("{}");
        await Assert.ThrowsAsync<NotSupportedException>(
            () => provider.EmbedAsync(
                new ProviderEmbeddingRequest { ModelApiName = "test", Input = "test" },
                CancellationToken.None));
    }

    [Fact]
    public async Task GenerateImageAsync_ThrowsNotSupported()
    {
        var provider = CreateProvider("{}");
        await Assert.ThrowsAsync<NotSupportedException>(
            () => provider.GenerateImageAsync(
                new ProviderImageRequest { Prompt = "test" },
                CancellationToken.None));
    }

    [Fact]
    public async Task CompleteAsync_WithAttachment_UsesAssistantsPurposeForUpload()
    {
        string? uploadBody = null;
        var handler = new MockHttpHandler(async req =>
        {
            var body = req.Content is null ? null : await req.Content.ReadAsStringAsync();

            if (req.RequestUri!.AbsolutePath == "/v1/files" && req.Method == HttpMethod.Post)
            {
                uploadBody = body;
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"id":"file_xai"}""", System.Text.Encoding.UTF8, "application/json"),
                };
            }

            if (req.RequestUri!.AbsolutePath == "/v1/responses" && req.Method == HttpMethod.Post)
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""
                    {
                        "status": "completed",
                        "output": [{
                            "type": "message",
                            "content": [{ "type": "output_text", "text": "done" }]
                        }],
                        "usage": { "input_tokens": 10, "output_tokens": 2 }
                    }
                    """, System.Text.Encoding.UTF8, "application/json"),
                };
            }

            if (req.RequestUri!.AbsolutePath == "/v1/files/file_xai" && req.Method == HttpMethod.Delete)
            {
                return new HttpResponseMessage(HttpStatusCode.OK);
            }

            throw new InvalidOperationException($"Unexpected request: {req.Method} {req.RequestUri}");
        });

        var provider = new XaiProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://api.x.ai",
            ApiKey = "test",
        });

        await provider.CompleteAsync(new ProviderRequest
        {
            ModelApiName = "grok-4",
            Messages =
            [
                new ChatMessage
                {
                    Role = ChatRole.User,
                    Content = "Read the attachment",
                    Attachments =
                    [
                        new DocumentAttachment
                        {
                            FileName = "notes.txt",
                            MimeType = "text/plain",
                            Content = System.Text.Encoding.UTF8.GetBytes("Status: GREEN"),
                        }
                    ],
                }
            ],
        }, CancellationToken.None);

        Assert.NotNull(uploadBody);
        Assert.Contains("assistants", uploadBody!, StringComparison.Ordinal);
    }

    private static XaiProvider CreateProvider(string responseJson)
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var httpClient = new HttpClient(handler);
        var context = new ProviderRequestContext
        {
            HttpClient = httpClient,
            BaseUrl = "https://api.x.ai",
            ApiKey = "test",
        };

        return new XaiProvider(context);
    }

    private static ProviderRequest BuildSimpleRequest() => new()
    {
        ModelApiName = "grok-4",
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    };

    private static string BuildChatResponse(string content) => $$"""
    {
        "choices": [{
            "message": { "content": "{{content}}" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
    }
    """;
}
