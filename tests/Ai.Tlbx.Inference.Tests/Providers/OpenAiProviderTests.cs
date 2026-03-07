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

    [Fact]
    public async Task CompleteAsync_WithAttachment_UploadsFileAndUsesResponsesApi()
    {
        var requests = new List<(string Method, string Url, string? Body)>();
        var handler = new MockHttpHandler(async req =>
        {
            var body = req.Content is null ? null : await req.Content.ReadAsStringAsync();
            requests.Add((req.Method.Method, req.RequestUri!.ToString(), body));

            if (req.RequestUri!.AbsolutePath == "/v1/files" && req.Method == HttpMethod.Post)
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"id":"file_123"}""", System.Text.Encoding.UTF8, "application/json"),
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

            if (req.RequestUri!.AbsolutePath == "/v1/files/file_123" && req.Method == HttpMethod.Delete)
            {
                return new HttpResponseMessage(HttpStatusCode.OK);
            }

            throw new InvalidOperationException($"Unexpected request: {req.Method} {req.RequestUri}");
        });

        var provider = new OpenAiProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://api.openai.com",
            ApiKey = "test-key",
        });

        var response = await provider.CompleteAsync(new ProviderRequest
        {
            ModelApiName = "gpt-5",
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

        Assert.Equal("done", response.Content);
        Assert.Contains(requests, request => request.Url.EndsWith("/v1/files", StringComparison.Ordinal));
        Assert.Contains(requests, request => request.Url.EndsWith("/v1/responses", StringComparison.Ordinal));
        Assert.Contains(requests, request => request.Url.EndsWith("/v1/files/file_123", StringComparison.Ordinal));
        Assert.Contains(requests, request => request.Body?.Contains("\"input_file\"", StringComparison.Ordinal) == true);
    }

    [Fact]
    public async Task StreamAsync_ResponsesApiModel_UsesResponsesEndpoint()
    {
        HttpRequestMessage? captured = null;
        var sseData = """
        data: {"type":"response.output_text.delta","delta":"Hello "}

        data: {"type":"response.output_text.delta","delta":"world"}

        data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2}}}

        """;

        var handler = new MockHttpHandler(async req =>
        {
            captured = req;
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sseData, System.Text.Encoding.UTF8, "text/event-stream"),
            };
        });

        var provider = new OpenAiProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://api.openai.com",
            ApiKey = "test-key",
        });

        var chunks = new List<string>();
        await foreach (var evt in provider.StreamAsync(new ProviderRequest
        {
            ModelApiName = "gpt-5.2-pro",
            PreferredEndpoint = ModelEndpointFamily.Responses,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
        }, CancellationToken.None))
        {
            if (evt.TextDelta is not null)
            {
                chunks.Add(evt.TextDelta);
            }
        }

        Assert.NotNull(captured);
        Assert.Equal("https://api.openai.com/v1/responses", captured!.RequestUri!.ToString());
        Assert.Equal(2, chunks.Count);
        Assert.Equal("Hello world", string.Concat(chunks));
    }

    [Fact]
    public async Task StreamAsync_ResponsesApi_EmitsCompletedTextWhenNoDeltaWasStreamed()
    {
        var sseData = """
        data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"Hello world from completed event"}]}],"usage":{"input_tokens":5,"output_tokens":6}}}

        """;

        var provider = new OpenAiProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(new MockHttpHandler(async _ =>
            {
                await Task.CompletedTask;
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(sseData, System.Text.Encoding.UTF8, "text/event-stream"),
                };
            })),
            BaseUrl = "https://api.openai.com",
            ApiKey = "test-key",
        });

        var chunks = new List<string>();
        await foreach (var evt in provider.StreamAsync(new ProviderRequest
        {
            ModelApiName = "gpt-5",
            PreferredEndpoint = ModelEndpointFamily.Responses,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
        }, CancellationToken.None))
        {
            if (evt.TextDelta is not null)
            {
                chunks.Add(evt.TextDelta);
            }
        }

        Assert.Single(chunks);
        Assert.Equal("Hello world from completed event", chunks[0]);
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
