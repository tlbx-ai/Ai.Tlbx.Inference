using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class AnthropicProviderTests
{
    [Fact]
    public async Task CompleteAsync_SendsXApiKeyHeader()
    {
        HttpRequestMessage? captured = null;
        var json = BuildAnthropicResponse("Hello");
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "sk-ant-test",
        };

        var provider = new AnthropicProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(captured);
        Assert.True(captured!.Headers.TryGetValues("x-api-key", out var values));
        Assert.Equal("sk-ant-test", values!.First());
    }

    [Fact]
    public async Task CompleteAsync_SendsAnthropicVersionHeader()
    {
        HttpRequestMessage? captured = null;
        var json = BuildAnthropicResponse("Hello");
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        };

        var provider = new AnthropicProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(captured);
        Assert.True(captured!.Headers.TryGetValues("anthropic-version", out var values));
        Assert.Equal("2023-06-01", values!.First());
    }

    [Fact]
    public async Task CompleteAsync_ParsesContent()
    {
        var json = BuildAnthropicResponse("Hello from Claude!");
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal("Hello from Claude!", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_ParsesUsage()
    {
        var json = """
        {
            "content": [{ "type": "text", "text": "Hi" }],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 25,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5
            },
            "stop_reason": "end_turn"
        }
        """;
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal(50, response.Usage.InputTokens);
        Assert.Equal(25, response.Usage.OutputTokens);
        Assert.Equal(10, response.Usage.CacheReadTokens);
        Assert.Equal(5, response.Usage.CacheWriteTokens);
    }

    [Fact]
    public async Task CompleteAsync_SendsSystemMessageInTopLevelField()
    {
        string? capturedBody = null;
        var json = BuildAnthropicResponse("Hello");
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        };

        var provider = new AnthropicProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "claude-opus-4-6",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
            SystemMessage = "You are helpful.",
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        Assert.Equal("You are helpful.", doc.RootElement.GetProperty("system").GetString());
    }

    [Fact]
    public async Task CompleteAsync_EnableCache_SendsSystemWithCacheControl()
    {
        string? capturedBody = null;
        var json = BuildAnthropicResponse("Hello");
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        };

        var provider = new AnthropicProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "claude-opus-4-6",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
            SystemMessage = "Cached system prompt.",
            EnableCache = true,
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var system = doc.RootElement.GetProperty("system");
        Assert.Equal(JsonValueKind.Array, system.ValueKind);
        var first = system[0];
        Assert.Equal("text", first.GetProperty("type").GetString());
        Assert.Equal("Cached system prompt.", first.GetProperty("text").GetString());
        Assert.Equal("ephemeral", first.GetProperty("cache_control").GetProperty("type").GetString());
    }

    [Fact]
    public async Task CompleteAsync_ThinkingBudget_SendsThinkingConfig()
    {
        string? capturedBody = null;
        var json = BuildAnthropicResponse("Thought result");
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        };

        var provider = new AnthropicProvider(context);
        var request = new ProviderRequest
        {
            ModelApiName = "claude-opus-4-6",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Think!" }],
            ThinkingBudget = 10000,
        };

        await provider.CompleteAsync(request, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var thinking = doc.RootElement.GetProperty("thinking");
        Assert.Equal("enabled", thinking.GetProperty("type").GetString());
        Assert.Equal(10000, thinking.GetProperty("budget_tokens").GetInt32());
    }

    [Fact]
    public async Task CompleteAsync_ParsesToolCalls()
    {
        var json = """
        {
            "content": [
                { "type": "text", "text": "Let me check." },
                { "type": "tool_use", "id": "tu_123", "name": "search", "input": {"query": "test"} }
            ],
            "usage": { "input_tokens": 10, "output_tokens": 20 },
            "stop_reason": "tool_use"
        }
        """;
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.Equal("Let me check.", response.Content);
        Assert.NotNull(response.ToolCalls);
        Assert.Single(response.ToolCalls);
        Assert.Equal("tu_123", response.ToolCalls[0].Id);
        Assert.Equal("search", response.ToolCalls[0].Name);
    }

    [Fact]
    public async Task CompleteAsync_SendsToCorrectEndpoint()
    {
        HttpRequestMessage? captured = null;
        var json = BuildAnthropicResponse("Hello");
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        };

        var provider = new AnthropicProvider(context);
        await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(captured);
        Assert.Equal("https://api.anthropic.com/v1/messages", captured!.RequestUri?.ToString());
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

    private static AnthropicProvider CreateProvider(string responseJson)
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
            BaseUrl = "https://api.anthropic.com",
            ApiKey = "test",
        };

        return new AnthropicProvider(context);
    }

    private static ProviderRequest BuildSimpleRequest() => new()
    {
        ModelApiName = "claude-opus-4-6",
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    };

    private static string BuildAnthropicResponse(string text) => $$"""
    {
        "content": [{ "type": "text", "text": "{{text}}" }],
        "usage": { "input_tokens": 10, "output_tokens": 5 },
        "stop_reason": "end_turn"
    }
    """;
}
