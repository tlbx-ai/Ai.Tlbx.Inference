using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Client;

public sealed class AiInferenceClientTests
{
    [Fact]
    public async Task CompleteAsync_RoutesToOpenAi()
    {
        HttpRequestMessage? captured = null;
        var handler = new MockHttpHandler(async req =>
        {
            captured = req;
            await Task.CompletedTask;
            return BuildOpenAiResponse("Hello from OpenAI");
        });

        var client = CreateClient(handler, o => o.AddOpenAi("sk-test"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var response = await client.CompleteAsync(request);

        Assert.Equal("Hello from OpenAI", response.Content);
        Assert.NotNull(captured);
        Assert.Contains("api.openai.com", captured!.RequestUri?.Host ?? "");
    }

    [Fact]
    public async Task CompleteAsync_RoutesToAnthropic()
    {
        HttpRequestMessage? captured = null;
        var handler = new MockHttpHandler(async req =>
        {
            captured = req;
            await Task.CompletedTask;
            return BuildAnthropicResponse("Hello from Claude");
        });

        var client = CreateClient(handler, o => o.AddAnthropic("sk-ant-test"));
        var request = new CompletionRequest
        {
            Model = AiModel.ClaudeOpus46,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var response = await client.CompleteAsync(request);

        Assert.Equal("Hello from Claude", response.Content);
        Assert.NotNull(captured);
        Assert.Contains("api.anthropic.com", captured!.RequestUri?.Host ?? "");
    }

    [Fact]
    public async Task CompleteAsync_RoutesToGoogle()
    {
        HttpRequestMessage? captured = null;
        var handler = new MockHttpHandler(async req =>
        {
            captured = req;
            await Task.CompletedTask;
            return BuildGoogleResponse("Hello from Gemini");
        });

        var client = CreateClient(handler, o => o.AddGoogle("test-key"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gemini31FlashLitePreview,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var response = await client.CompleteAsync(request);

        Assert.Equal("Hello from Gemini", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_RoutesToXai()
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return BuildOpenAiResponse("Hello from Grok");
        });

        var client = CreateClient(handler, o => o.AddXai("xai-test"));
        var request = new CompletionRequest
        {
            Model = AiModel.Grok4,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var response = await client.CompleteAsync(request);

        Assert.Equal("Hello from Grok", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_ThrowsWhenProviderNotConfigured()
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return BuildOpenAiResponse("nope");
        });

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var request = new CompletionRequest
        {
            Model = AiModel.ClaudeOpus46,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.CompleteAsync(request));
    }

    [Fact]
    public async Task CompleteAsync_Generic_DeserializesJson()
    {
        var json = """
        {
            "choices": [{
                "message": { "content": "{\"name\":\"Alice\",\"age\":30}" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 20 }
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

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            JsonSchema = """{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}""",
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Give me a person" }],
        };

        var response = await client.CompleteAsync(request, AiInferenceClientTestJsonContext.Default.TestPerson);

        Assert.Equal("Alice", response.Content.Name);
        Assert.Equal(30, response.Content.Age);
        Assert.Equal(AiModel.Gpt52, response.Model);
    }

    [Fact]
    public async Task CompleteAsync_Generic_StringDelegatesToNonGeneric()
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return BuildOpenAiResponse("plain text");
        });

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var response = await client.CompleteAsync(request);

        Assert.Equal("plain text", response.Content);
    }

    [Fact]
    public async Task CompleteAsync_SetsModelOnResponse()
    {
        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return BuildOpenAiResponse("test");
        });

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt54,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var response = await client.CompleteAsync(request);

        Assert.Equal(AiModel.Gpt54, response.Model);
    }

    [Fact]
    public async Task StreamAsync_YieldsTextDeltas()
    {
        var sseData = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n" +
                      "data: {\"choices\":[{\"delta\":{\"content\":\" World\"}}]}\n\n" +
                      "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n" +
                      "data: [DONE]\n\n";

        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sseData, System.Text.Encoding.UTF8, "text/event-stream"),
            };
        });

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var chunks = new List<string>();
        await foreach (var chunk in client.StreamAsync(request))
        {
            chunks.Add(chunk);
        }

        Assert.Equal(2, chunks.Count);
        Assert.Equal("Hello", chunks[0]);
        Assert.Equal(" World", chunks[1]);
    }

    [Fact]
    public async Task StreamAsync_SplitsLargeTextDeltas()
    {
        var longText = string.Join(" ", Enumerable.Repeat("chunk", 80));
        var sseData = $"data: {{\"choices\":[{{\"delta\":{{\"content\":{JsonSerializer.Serialize(longText)}}}}}]}}\n\n" +
                      "data: [DONE]\n\n";

        var handler = new MockHttpHandler(async _ =>
        {
            await Task.CompletedTask;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(sseData, System.Text.Encoding.UTF8, "text/event-stream"),
            };
        });

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
        };

        var chunks = new List<string>();
        await foreach (var chunk in client.StreamAsync(request))
        {
            chunks.Add(chunk);
        }

        Assert.True(chunks.Count > 1);
        Assert.Equal(longText, string.Concat(chunks));
    }

    [Fact]
    public async Task CompleteAsync_SnapshotsCallerOwnedMessagesAndAttachments()
    {
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        string? capturedBody = null;
        string? uploadBody = null;

        var handler = new MockHttpHandler(async req =>
        {
            await gate.Task;
            if (req.RequestUri!.AbsolutePath == "/v1/files")
            {
                uploadBody = await req.Content!.ReadAsStringAsync();
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"id":"file_123"}""", System.Text.Encoding.UTF8, "application/json"),
                };
            }

            if (req.RequestUri!.AbsolutePath == "/v1/responses")
            {
                capturedBody = await req.Content!.ReadAsStringAsync();
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""
                    {
                        "status": "completed",
                        "output": [{
                            "type": "message",
                            "content": [{ "type": "output_text", "text": "ok" }]
                        }],
                        "usage": { "input_tokens": 10, "output_tokens": 2 }
                    }
                    """, System.Text.Encoding.UTF8, "application/json"),
                };
            }

            if (req.RequestUri!.AbsolutePath == "/v1/files/file_123")
            {
                return new HttpResponseMessage(HttpStatusCode.OK);
            }

            throw new InvalidOperationException($"Unexpected request: {req.Method} {req.RequestUri}");
        });

        var attachmentBytes = System.Text.Encoding.UTF8.GetBytes("original attachment");
        var messages = new List<ChatMessage>
        {
            new()
            {
                Role = ChatRole.User,
                Content = "original prompt",
                Attachments =
                [
                    new DocumentAttachment
                    {
                        FileName = "note.txt",
                        MimeType = "text/plain",
                        Content = attachmentBytes,
                    }
                ],
            }
        };

        var request = new CompletionRequest
        {
            Model = AiModel.Gpt52,
            Messages = messages,
            StopSequences = ["STOP"],
        };

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var task = client.CompleteAsync(request);

        messages[0] = messages[0] with { Content = "mutated prompt" };
        attachmentBytes[0] = (byte)'X';
        gate.SetResult();

        var response = await task;

        Assert.Equal("ok", response.Content);
        Assert.NotNull(capturedBody);
        Assert.NotNull(uploadBody);
        Assert.Contains("original prompt", capturedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("mutated prompt", capturedBody, StringComparison.Ordinal);
        Assert.Contains("\"file_id\":\"file_123\"", capturedBody, StringComparison.Ordinal);
        Assert.Contains("original attachment", uploadBody, StringComparison.Ordinal);
        Assert.DoesNotContain(System.Text.Encoding.UTF8.GetString(attachmentBytes), uploadBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CompleteWithToolsAsync_SnapshotsCallerOwnedToolDefinitions()
    {
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        string? capturedBody = null;

        var firstResponse = """
        {
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"Berlin\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        }
        """;

        var finalResponse = BuildOpenAiResponse("done");
        var callCount = 0;
        var handler = new MockHttpHandler(async req =>
        {
            callCount++;
            if (callCount == 1)
            {
                await gate.Task;
                capturedBody = await req.Content!.ReadAsStringAsync();
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(firstResponse, System.Text.Encoding.UTF8, "application/json"),
                };
            }

            return finalResponse;
        });

        var toolSchema = JsonDocument.Parse("""{"type":"object"}""").RootElement.Clone();
        var tools = new List<ToolDefinition>
        {
            new()
            {
                Name = "lookup_weather",
                Description = "original description",
                ParametersSchema = toolSchema,
            }
        };

        var client = CreateClient(handler, o => o.AddOpenAi("key"));
        var task = client.CompleteWithToolsAsync(
            new CompletionRequest
            {
                Model = AiModel.Gpt52,
                Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hi" }],
            },
            tools,
            toolCall => Task.FromResult(new ToolCallResult
            {
                ToolCallId = toolCall.Id,
                Result = "{\"ok\":true}",
            }));

        tools[0] = tools[0] with
        {
            Name = "mutated_tool",
            Description = "mutated description",
        };
        gate.SetResult();

        var response = await task;

        Assert.Equal("done", response.Content);
        Assert.NotNull(capturedBody);
        Assert.Contains("lookup_weather", capturedBody, StringComparison.Ordinal);
        Assert.Contains("original description", capturedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("mutated_tool", capturedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("mutated description", capturedBody, StringComparison.Ordinal);
    }

    private static AiInferenceClient CreateClient(MockHttpHandler handler, Action<AiInferenceOptions> configure)
    {
        var httpClient = new HttpClient(handler);
        var options = new AiInferenceOptions();
        configure(options);
        return new AiInferenceClient(httpClient, options);
    }

    private static HttpResponseMessage BuildOpenAiResponse(string content) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent($$"""
            {
                "choices": [{ "message": { "content": "{{content}}" }, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
            }
            """, System.Text.Encoding.UTF8, "application/json"),
        };

    private static HttpResponseMessage BuildAnthropicResponse(string content) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent($$"""
            {
                "content": [{ "type": "text", "text": "{{content}}" }],
                "usage": { "input_tokens": 10, "output_tokens": 5 },
                "stop_reason": "end_turn"
            }
            """, System.Text.Encoding.UTF8, "application/json"),
        };

    private static HttpResponseMessage BuildGoogleResponse(string content) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent($$"""
            {
                "candidates": [{ "content": { "parts": [{ "text": "{{content}}" }] } }],
                "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5 }
            }
            """, System.Text.Encoding.UTF8, "application/json"),
        };

    public sealed class TestPerson
    {
        public string Name { get; set; } = "";
        public int Age { get; set; }
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(AiInferenceClientTests.TestPerson))]
internal sealed partial class AiInferenceClientTestJsonContext : JsonSerializerContext
{
}
