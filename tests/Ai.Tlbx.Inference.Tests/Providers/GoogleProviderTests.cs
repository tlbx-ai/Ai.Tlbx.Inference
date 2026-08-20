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
        Assert.Equal(60, response.Usage.OutputTokens);
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

    [Fact]
    public async Task CompleteAsync_ParsesToolCallThoughtSignature()
    {
        var json = """
        {
            "candidates": [{
                "content": {
                    "parts": [
                        {
                            "functionCall": { "name": "get_weather", "args": {"city": "Berlin"} },
                            "thoughtSignature": "gemini-signature"
                        }
                    ]
                }
            }]
        }
        """;
        var provider = CreateProvider(json);

        var response = await provider.CompleteAsync(BuildSimpleRequest(), CancellationToken.None);

        Assert.NotNull(response.ToolCalls);
        var toolCall = Assert.Single(response.ToolCalls);
        Assert.Equal("gemini-signature", toolCall.ThoughtSignature);
    }

    [Fact]
    public async Task CompleteAsync_SendsToolCallThoughtSignatureBackToGemini()
    {
        string? capturedBody = null;
        var handler = new MockHttpHandler(async req =>
        {
            capturedBody = await req.Content!.ReadAsStringAsync();
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(BuildGoogleResponse("Done"), System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var provider = new GoogleProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        });

        await provider.CompleteAsync(new ProviderRequest
        {
            ModelApiName = "gemini-2.5-pro",
            Messages =
            [
                new ChatMessage
                {
                    Role = ChatRole.Assistant,
                    ToolCalls =
                    [
                        new ToolCallRequest
                        {
                            Id = "call-1",
                            Name = "get_weather",
                            Arguments = """{"city":"Berlin"}""",
                            ThoughtSignature = "gemini-signature",
                        }
                    ],
                },
            ],
        }, CancellationToken.None);

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var part = doc.RootElement.GetProperty("contents")[0].GetProperty("parts")[0];
        Assert.Equal("gemini-signature", part.GetProperty("thoughtSignature").GetString());
    }

    [Fact]
    public async Task CompleteAsync_WithTextAttachment_SendsInlineData()
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

        var provider = new GoogleProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        });

        await provider.CompleteAsync(new ProviderRequest
        {
            ModelApiName = "gemini-2.5-pro",
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

        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var parts = doc.RootElement.GetProperty("contents")[0].GetProperty("parts");
        Assert.Equal("text/plain", parts[0].GetProperty("inline_data").GetProperty("mime_type").GetString());
    }

    [Fact]
    public async Task StreamAsync_YieldsMultipleTextDeltas()
    {
        var sse = """
        data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]} }]}

        data: {"candidates":[{"content":{"parts":[{"text":"Gemini"}]} }],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}

        """;

        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sse, System.Text.Encoding.UTF8, "text/event-stream"),
            };
        });

        var provider = new GoogleProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        });

        var chunks = new List<string>();
        await foreach (var evt in provider.StreamAsync(BuildSimpleRequest(), CancellationToken.None))
        {
            if (evt.TextDelta is not null)
            {
                chunks.Add(evt.TextDelta);
            }
        }

        Assert.Equal(2, chunks.Count);
        Assert.Equal("Hello Gemini", string.Concat(chunks));
    }

    [Fact]
    public async Task GenerateImageAsync_SendsImageModalityAndParsesInlineData()
    {
        string? capturedBody = null;
        HttpRequestMessage? capturedRequest = null;
        var expectedBytes = "png-bytes"u8.ToArray();
        var responseJson = $$"""
        {
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "Generated image." },
                        { "inlineData": { "data": "{{Convert.ToBase64String(expectedBytes)}}", "mimeType": "image/png" } }
                    ]
                }
            }]
        }
        """;

        var handler = new MockHttpHandler(async req =>
        {
            capturedRequest = req;
            capturedBody = await req.Content!.ReadAsStringAsync();
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, System.Text.Encoding.UTF8, "application/json"),
            };
        });

        var provider = new GoogleProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(handler),
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        });

        var bytes = await provider.GenerateImageAsync(new ProviderImageRequest
        {
            ModelApiName = "gemini-2.5-flash-image",
            Prompt = "Generate a skyline",
        }, CancellationToken.None);

        Assert.Equal(expectedBytes, bytes);
        Assert.NotNull(capturedRequest);
        Assert.Contains("gemini-2.5-flash-image:generateContent", capturedRequest!.RequestUri!.ToString());
        Assert.NotNull(capturedBody);
        using var doc = JsonDocument.Parse(capturedBody!);
        var generationConfig = doc.RootElement.GetProperty("generationConfig");
        Assert.False(generationConfig.TryGetProperty("responseMimeType", out _));
        var responseModalities = generationConfig.GetProperty("responseModalities");
        Assert.Equal("IMAGE", responseModalities[0].GetString());
    }

    [Fact]
    public async Task CompleteAsync_Grounding_RequestsGoogleSearchAndParsesBilledQueries()
    {
        string? capturedBody = null;
        var responseJson = """
        {
          "candidates":[{
            "content":{"parts":[{"text":"Aktuelle Antwort"}]},
            "groundingMetadata":{
              "webSearchQueries":["aktuelle bildung nachrichten","bildung 2026"],
              "groundingChunks":[{"web":{"uri":"https://example.org/current","title":"Current source"}}]
            }
          }],
          "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":10,"billedToolCalls":[{"tool":"GOOGLE_SEARCH_RETRIEVAL","successfulToolCallCount":2}]}
        }
        """;
        var provider = new GoogleProvider(new ProviderRequestContext
        {
            HttpClient = new HttpClient(new MockHttpHandler(async request =>
            {
                capturedBody = await request.Content!.ReadAsStringAsync();
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(responseJson, System.Text.Encoding.UTF8, "application/json"),
                };
            })),
            BaseUrl = "https://generativelanguage.googleapis.com",
            ApiKey = "test",
        });

        var response = await provider.CompleteAsync(new ProviderRequest
        {
            ModelApiName = "gemini-3.5-flash",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Was ist aktuell?" }],
            Grounding = new GroundingOptions(),
        }, CancellationToken.None);

        using var request = JsonDocument.Parse(capturedBody!);
        Assert.True(request.RootElement.GetProperty("tools")[0].TryGetProperty("googleSearch", out _));
        Assert.NotNull(response.Grounding);
        Assert.Equal(2, response.Grounding!.SearchQueries.Count);
        Assert.Single(response.Grounding.Sources);
        Assert.Equal(2, response.Grounding.Usage.WebSearchCalls);
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
