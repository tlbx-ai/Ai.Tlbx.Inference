using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class OpenAiProviderTests
{
    private static (OpenAiProvider Provider, HttpRequestMessage? CapturedRequest) CreateProvider(
        string responseJson,
        HttpStatusCode statusCode = HttpStatusCode.OK)
    {
        HttpRequestMessage? captured = null;
        var handler = new MockHttpHandler(async req =>
        {
            captured = req;
            await Task.CompletedTask;
            return new HttpResponseMessage(statusCode)
            {
                Content = new StringContent(responseJson, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var httpClient = new HttpClient(handler);
        var context = new ProviderRequestContext
        {
            HttpClient = httpClient,
            BaseUrl = "https://api.openai.com",
            ApiKey = "test-key",
        };

        return (new OpenAiProvider(context), captured);
    }

    [Fact]
    public async Task CompleteAsync_SendsCorrectUrl()
    {
        var json = BuildChatResponse("Hello!", 10, 5);
        var (provider, _) = CreateProvider(json);

        var request = BuildSimpleRequest();
        var response = await provider.CompleteAsync(request, CancellationToken.None);

        Assert.Equal("Hello!", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_SetsAuthorizationHeader()
    {
        var json = BuildChatResponse("Hi", 10, 5);
        HttpRequestMessage? captured = null;
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
            BaseUrl = "https://api.openai.com",
            ApiKey = "sk-test-123",
        };

        var provider = new OpenAiProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(captured);
        Assert.Equal("Bearer", captured!.Headers.Authorization?.Scheme);
        Assert.Equal("sk-test-123", captured.Headers.Authorization?.Parameter);
    }

    [Fact]
    public async Task CompleteAsync_ParsesTokenUsage()
    {
        var json = BuildChatResponse("Hello", 100, 50, cacheRead: 20, thinking: 10);
        var (provider, _) = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal(100, response.Usage.InputTokens);
        Assert.Equal(50, response.Usage.OutputTokens);
        Assert.Equal(20, response.Usage.CacheReadTokens);
        Assert.Equal(10, response.Usage.ThinkingTokens);
    }

    [Fact]
    public async Task CompleteAsync_ParsesToolCalls()
    {
        var json = """
        {
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"London\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        }
        """;
        var (provider, _) = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(response.ToolCalls);
        Assert.Single(response.ToolCalls);
        Assert.Equal("call_123", response.ToolCalls[0].Id);
        Assert.Equal("get_weather", response.ToolCalls[0].Name);
        Assert.Equal("{\"city\":\"London\"}", response.ToolCalls[0].Arguments);
    }

    [Fact]
    public async Task CompleteAsync_ThrowsOnError()
    {
        var (provider, _) = CreateProvider("{\"error\":\"bad\"}", HttpStatusCode.BadRequest);

        await Assert.ThrowsAsync<HttpRequestException>(
            () => provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None));
    }

    [Fact]
    public async Task CompleteAsync_SendsCorrectRequestPath()
    {
        var json = BuildChatResponse("Hi", 1, 1);
        HttpRequestMessage? captured = null;
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
            BaseUrl = "https://api.openai.com",
            ApiKey = "test",
        };

        var provider = new OpenAiProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(captured);
        Assert.Equal("https://api.openai.com/v1/chat/completions", captured!.RequestUri?.ToString());
    }

    [Fact]
    public async Task CompleteAsync_SendsModelInBody()
    {
        var json = BuildChatResponse("Hi", 1, 1);
        string? capturedBody = null;
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
            BaseUrl = "https://api.openai.com",
            ApiKey = "test",
        };

        var provider = new OpenAiProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest("gpt-5.2"), CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        Assert.Equal("gpt-5.2", doc.RootElement.GetProperty("model").GetString());
    }

    private static ProviderRequest BuildSimpleRequest(string model = "gpt-5.2") => new()
    {
        ModelApiName = model,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    };

    private static string BuildChatResponse(
        string content,
        int promptTokens,
        int completionTokens,
        int cacheRead = 0,
        int thinking = 0)
    {
        var promptDetails = cacheRead > 0 ? $",\"prompt_tokens_details\":{{\"cached_tokens\":{cacheRead}}}" : "";
        var completionDetails = thinking > 0 ? $",\"completion_tokens_details\":{{\"reasoning_tokens\":{thinking}}}" : "";

        return $$"""
        {
            "choices": [{
                "message": { "content": "{{content}}" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": {{promptTokens}},
                "completion_tokens": {{completionTokens}}
                {{promptDetails}}
                {{completionDetails}}
            }
        }
        """;
    }
}
