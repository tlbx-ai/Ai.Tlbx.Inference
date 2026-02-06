using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class GoogleProviderTests
{
    [Fact]
    public async Task CompleteAsync_AiStudio_SendsCorrectUrl()
    {
        HttpRequestMessage? captured = null;
        var json = BuildGoogleResponse("Hello");
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
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test-api-key",
        };

        var provider = new GoogleProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest("gemini-2.5-pro"), CancellationToken.None);

        Assert.NotNull(captured);
        var url = captured!.RequestUri?.ToString() ?? "";
        Assert.Contains("generativelanguage.googleapis.com", url);
        Assert.Contains("gemini-2.5-pro:generateContent", url);
        Assert.Contains("key=test-api-key", url);
    }

    [Fact]
    public async Task CompleteAsync_ParsesContent()
    {
        var json = BuildGoogleResponse("Hello from Gemini!");
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal("Hello from Gemini!", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_ParsesUsage()
    {
        var json = """
        {
            "candidates": [{
                "content": { "parts": [{ "text": "Hi" }] }
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "cachedContentTokenCount": 20,
                "thoughtsTokenCount": 10
            }
        }
        """;
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal(100, response.Usage.InputTokens);
        Assert.Equal(50, response.Usage.OutputTokens);
        Assert.Equal(20, response.Usage.CacheReadTokens);
        Assert.Equal(10, response.Usage.ThinkingTokens);
    }

    [Fact]
    public async Task CompleteAsync_SendsSystemInstruction()
    {
        string? capturedBody = null;
        var json = BuildGoogleResponse("Hi");
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
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        };

        var provider = new GoogleProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "gemini-2.5-pro",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
            SystemMessage = "Be concise.",
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var sysInstruction = doc.RootElement.GetProperty("system_instruction");
        var text = sysInstruction.GetProperty("parts")[0].GetProperty("text").GetString();
        Assert.Equal("Be concise.", text);
    }

    [Fact]
    public async Task CompleteAsync_ThinkingBudget_SendsThinkingConfig()
    {
        string? capturedBody = null;
        var json = BuildGoogleResponse("Thought");
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
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        };

        var provider = new GoogleProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "gemini-2.5-pro",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Think" }],
            ThinkingBudget = 8000,
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var genConfig = doc.RootElement.GetProperty("generationConfig");
        var thinkingConfig = genConfig.GetProperty("thinkingConfig");
        Assert.Equal(8000, thinkingConfig.GetProperty("thinkingBudget").GetInt32());
    }

    [Fact]
    public async Task CompleteAsync_ParsesToolCalls()
    {
        var json = """
        {
            "candidates": [{
                "content": {
                    "parts": [
                        { "functionCall": { "name": "get_weather", "args": {"city": "Berlin"} } }
                    ]
                }
            }],
            "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5 }
        }
        """;
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(response.ToolCalls);
        Assert.Single(response.ToolCalls);
        Assert.Equal("get_weather", response.ToolCalls[0].Name);
        Assert.Contains("Berlin", response.ToolCalls[0].Arguments);
    }

    private static GoogleProvider CreateProvider(string responseJson)
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
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        };

        return new GoogleProvider(context);
    }

    private static ProviderRequest BuildSimpleRequest(string model = "gemini-2.5-pro") => new()
    {
        ModelApiName = model,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    };

    private static string BuildGoogleResponse(string text) => $$"""
    {
        "candidates": [{
            "content": { "parts": [{ "text": "{{text}}" }] }
        }],
        "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5 }
    }
    """;
}
