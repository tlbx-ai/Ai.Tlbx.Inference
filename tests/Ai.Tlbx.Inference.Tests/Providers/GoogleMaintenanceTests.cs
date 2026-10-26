using System.Net;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Providers;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests.Providers;

public sealed class GoogleMaintenanceTests
{
    [Theory]
    [InlineData("gemini-4-argon", "low", "LOW")]
    [InlineData("gemini-3.8-flash", "medium", "MEDIUM")]
    [InlineData("gemini-3.5-flash-lite", "high", "HIGH")]
    public async Task Completion_UsesThinkingLevelForModernGemini(string model, string effort, string level)
    {
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var config = json.RootElement.GetProperty("generationConfig").GetProperty("thinkingConfig");
            Assert.Equal(level, config.GetProperty("thinkingLevel").GetString());
            Assert.False(config.TryGetProperty("thinkingBudget", out _));
            return JsonResponse("""{"candidates":[{"content":{"parts":[{"text":"OK"}]}}]}""");
        }));
        await CreateProvider(http).CompleteAsync(Request(model) with { ReasoningEffort = effort, ThinkingBudget = 16000 }, CancellationToken.None);
    }

    [Theory]
    [InlineData(1024, "LOW")]
    [InlineData(8192, "MEDIUM")]
    [InlineData(16000, "HIGH")]
    public async Task Completion_MapsLegacyBudgetToModernThinkingLevel(int budget, string level)
    {
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var config = json.RootElement.GetProperty("generationConfig").GetProperty("thinkingConfig");
            Assert.Equal(level, config.GetProperty("thinkingLevel").GetString());
            Assert.False(config.TryGetProperty("thinkingBudget", out _));
            return JsonResponse("""{"candidates":[{"content":{"parts":[{"text":"OK"}]}}]}""");
        }));
        await CreateProvider(http).CompleteAsync(Request("gemini-4-argon") with { ThinkingBudget = budget }, CancellationToken.None);
    }

    [Theory]
    [InlineData("minimal")]
    [InlineData("none")]
    [InlineData("max")]
    public async Task Completion_RejectsUnsupportedThinkingLevelBeforeHttp(string effort)
    {
        using var http = new HttpClient(new MockHttpHandler(_ => throw new InvalidOperationException("Unexpected HTTP")));
        await Assert.ThrowsAsync<ArgumentException>(() => CreateProvider(http).CompleteAsync(
            Request("gemini-3.8-flash") with { ReasoningEffort = effort }, CancellationToken.None));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Completion_RejectsArgonSamplingBeforeHttp(bool temperature)
    {
        using var http = new HttpClient(new MockHttpHandler(_ => throw new InvalidOperationException("Unexpected HTTP")));
        await Assert.ThrowsAsync<ArgumentException>(() => CreateProvider(http).CompleteAsync(
            Request("gemini-4-argon") with { Temperature = temperature ? 0.5 : null, TopP = temperature ? null : 0.9 }, CancellationToken.None));
    }

    [Fact]
    public async Task Completion_SendsJsonSchemaWithoutOpenApiSchemaConversion()
    {
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{"value":{"anyOf":[{"type":"string"},{"type":"null"}]}},"additionalProperties":false}""");
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var config = json.RootElement.GetProperty("generationConfig");
            Assert.Equal("application/json", config.GetProperty("responseMimeType").GetString());
            Assert.False(config.TryGetProperty("responseSchema", out _));
            Assert.False(config.GetProperty("responseJsonSchema").GetProperty("additionalProperties").GetBoolean());
            var declaration = json.RootElement.GetProperty("tools")[0].GetProperty("function_declarations")[0];
            Assert.True(declaration.TryGetProperty("parametersJsonSchema", out _));
            Assert.False(declaration.TryGetProperty("parameters", out _));
            return JsonResponse("""{"candidates":[{"content":{"parts":[{"text":"OK"}]}}]}""");
        }));
        await CreateProvider(http).CompleteAsync(Request("gemini-3.8-flash") with
        {
            JsonSchema = schema.RootElement.GetRawText(),
            Tools = [new ToolDefinition { Name = "extract", Description = "Extract", ParametersSchema = schema.RootElement }],
        }, CancellationToken.None);
    }

    [Theory]
    [InlineData("native-call-id")]
    [InlineData(null)]
    public async Task ToolRoundTrip_PreservesFunctionNameSignatureAndOptionalNativeId(string? nativeId)
    {
        var calls = 0;
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            calls++;
            if (calls == 1)
            {
                var responseJson = """{"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{}},"thoughtSignature":"opaque-signature"}]}}]}""";
                if (nativeId is not null)
                {
                    responseJson = responseJson.Replace("\"name\":\"lookup\"", $"\"id\":\"{nativeId}\",\"name\":\"lookup\"", StringComparison.Ordinal);
                }
                return JsonResponse(responseJson);
            }
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var contents = json.RootElement.GetProperty("contents");
            var callPart = contents[0].GetProperty("parts")[0];
            Assert.Equal("opaque-signature", callPart.GetProperty("thoughtSignature").GetString());
            var functionCall = callPart.GetProperty("functionCall");
            var functionResponse = contents[1].GetProperty("parts")[0].GetProperty("functionResponse");
            Assert.Equal("lookup", functionResponse.GetProperty("name").GetString());
            if (nativeId is not null)
            {
                Assert.Equal(nativeId, functionCall.GetProperty("id").GetString());
                Assert.Equal(nativeId, functionResponse.GetProperty("id").GetString());
            }
            else
            {
                Assert.False(functionCall.TryGetProperty("id", out _));
                Assert.False(functionResponse.TryGetProperty("id", out _));
            }
            return JsonResponse("""{"candidates":[{"content":{"parts":[{"text":"Done"}]}}]}""");
        }));
        var provider = CreateProvider(http);
        var first = await provider.CompleteAsync(Request("gemini-3.8-flash"), CancellationToken.None);
        var call = Assert.Single(first.ToolCalls!);
        Assert.Equal(nativeId, call.ProviderItemId);
        var result = await provider.CompleteAsync(Request("gemini-3.8-flash") with
        {
            Messages =
            [
                new ChatMessage { Role = ChatRole.Assistant, ToolCalls = first.ToolCalls },
                new ChatMessage { Role = ChatRole.Tool, ToolCallId = call.Id, Content = "42" },
            ],
        }, CancellationToken.None);
        Assert.Equal("Done", result.Content);
    }

    [Fact]
    public async Task Completion_ExcludesThoughtSummaryAndReportsContinuation()
    {
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(JsonResponse(
            """{"candidates":[{"finishReason":"CONTINUATION","content":{"parts":[{"thought":true,"text":"Internal summary"},{"text":"Visible"}]}}]}"""))));
        var response = await CreateProvider(http).CompleteAsync(Request("gemini-4-argon"), CancellationToken.None);
        Assert.Equal("Visible", response.Content);
        Assert.Contains("continuation", response.DiagnosticNote, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Streaming_ExcludesThoughtSummaryAndPreservesNativeToolIdentity()
    {
        const string sse = """
            data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"Internal"},{"text":"Visible"},{"functionCall":{"id":"call-1","name":"lookup","args":{}},"thoughtSignature":"signature"}]}}]}

            """;
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sse, System.Text.Encoding.UTF8, "text/event-stream"),
        })));
        var events = new List<ProviderStreamEvent>();
        await foreach (var evt in CreateProvider(http).StreamAsync(Request("gemini-3.8-flash"), CancellationToken.None)) events.Add(evt);
        Assert.Equal("Visible", string.Concat(events.Select(evt => evt.TextDelta)));
        var call = Assert.Single(events, evt => evt.ToolCall is not null).ToolCall!;
        Assert.Equal("call-1", call.Id);
        Assert.Equal("call-1", call.ProviderItemId);
        Assert.Equal("signature", call.ThoughtSignature);
    }

    [Theory]
    [InlineData(ImageGenerationModel.GeminiNanoBanana21, "gemini-nano-banana-2.1")]
    [InlineData(ImageGenerationModel.Gemini31FlashImage, "gemini-3.1-flash-image")]
    [InlineData(ImageGenerationModel.Gemini3ProImage, "gemini-3-pro-image")]
    public async Task FacadeImageEdit_RoutesMultipleReferenceImagesAndResolution(ImageGenerationModel model, string apiName)
    {
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            Assert.Equal($"/v1beta/models/{apiName}:generateContent", request.RequestUri!.AbsolutePath);
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var parts = json.RootElement.GetProperty("contents")[0].GetProperty("parts");
            Assert.Equal(3, parts.GetArrayLength());
            Assert.Equal("image/png", parts[0].GetProperty("inlineData").GetProperty("mimeType").GetString());
            Assert.Equal(Convert.ToBase64String("first-image"u8), parts[0].GetProperty("inlineData").GetProperty("data").GetString());
            Assert.Equal("image/jpeg", parts[1].GetProperty("inlineData").GetProperty("mimeType").GetString());
            Assert.Equal(Convert.ToBase64String("second-image"u8), parts[1].GetProperty("inlineData").GetProperty("data").GetString());
            Assert.Equal("Combine references", parts[2].GetProperty("text").GetString());
            var config = json.RootElement.GetProperty("generationConfig");
            Assert.Equal("IMAGE", config.GetProperty("responseModalities")[0].GetString());
            Assert.Equal("2K", config.GetProperty("imageConfig").GetProperty("imageSize").GetString());
            // Interim thought images must not be returned as the final edited result.
            return JsonResponse("""{"candidates":[{"content":{"parts":[{"thought":true,"inlineData":{"mimeType":"image/png","data":"dGhvdWdodA=="}},{"inlineData":{"mimeType":"image/png","data":"ZmluYWw="}}]}}]}""");
        }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddGoogle("test"));
        var bytes = await client.EditImageAsync(new ImageEditRequest
        {
            Model = model,
            Prompt = "Combine references",
            Size = "2K",
            Images =
            [
                new ImageEditInput { FileName = "first.png", MimeType = "image/png", Content = "first-image"u8.ToArray() },
                new ImageEditInput { FileName = "second.jpg", MimeType = "image/jpeg", Content = "second-image"u8.ToArray() },
            ],
        });
        Assert.Equal("final"u8.ToArray(), bytes);
    }

    [Theory]
    [InlineData("1024x1024", null)]
    [InlineData("2k", null)]
    [InlineData(null, "high")]
    public async Task ImageRequests_RejectUnsupportedSizeOrQualityBeforeHttp(string? size, string? quality)
    {
        using var http = new HttpClient(new MockHttpHandler(_ => throw new InvalidOperationException("Unexpected HTTP")));
        var provider = CreateProvider(http);
        await Assert.ThrowsAsync<ArgumentException>(() => provider.GenerateImageAsync(new ProviderImageRequest
        {
            ModelApiName = "gemini-nano-banana-2.1", Prompt = "Draw", Size = size, Quality = quality,
        }, CancellationToken.None));
        await Assert.ThrowsAsync<ArgumentException>(() => provider.EditImageAsync(new ProviderImageEditRequest
        {
            ModelApiName = "gemini-nano-banana-2.1", Prompt = "Edit", Size = size, Quality = quality,
            Images = [new ImageEditInput { FileName = "ref.png", MimeType = "image/png", Content = "ref"u8.ToArray() }],
        }, CancellationToken.None));
    }

    [Fact]
    public async Task ImageGeneration_ReportsRecitationRefusalWithoutRetry()
    {
        var calls = 0;
        using var http = new HttpClient(new MockHttpHandler(_ =>
        {
            calls++;
            return Task.FromResult(JsonResponse("""{"candidates":[{"finishReason":"IMAGE_RECITATION"}]}"""));
        }));
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => CreateProvider(http).GenerateImageAsync(
            new ProviderImageRequest { ModelApiName = "gemini-nano-banana-2.1", Prompt = "Draw" }, CancellationToken.None));
        Assert.Contains("Finish reason: IMAGE_RECITATION", exception.Message);
        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task ImageGeneration_DiagnosticsBoundVisibleTextAndExcludeThoughtImageData()
    {
        var responseJson = """{"promptFeedback":{"blockReason":"SAFETY"},"candidates":[{"finishReason":"SAFETY","content":{"parts":[{"thought":true,"text":"hidden reasoning","inlineData":{"data":"hidden-image-base64"}},{"text":"VISIBLE_TEXT"}]}}]}"""
            .Replace("VISIBLE_TEXT", new string('x', 500), StringComparison.Ordinal);
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(JsonResponse(responseJson))));
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => CreateProvider(http).GenerateImageAsync(
            new ProviderImageRequest { ModelApiName = "gemini-nano-banana-2.1", Prompt = "Draw" }, CancellationToken.None));
        Assert.Contains("Prompt block reason: SAFETY", exception.Message);
        Assert.Contains("Finish reason: SAFETY", exception.Message);
        Assert.EndsWith("Provider text: " + new string('x', 400), exception.Message);
        Assert.DoesNotContain("hidden", exception.Message);
        Assert.DoesNotContain("inlineData", exception.Message);
    }

    [Fact]
    public async Task Embedding2Batch_KeepsSeparateRequestsAndOrderedVectors()
    {
        using var http = new HttpClient(new MockHttpHandler(async request =>
        {
            Assert.Equal("/v1beta/models/gemini-embedding-2:batchEmbedContents", request.RequestUri!.AbsolutePath);
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync());
            var requests = json.RootElement.GetProperty("requests");
            Assert.Equal(2, requests.GetArrayLength());
            for (var i = 0; i < 2; i++)
            {
                Assert.Equal("models/gemini-embedding-2", requests[i].GetProperty("model").GetString());
                var parts = requests[i].GetProperty("content").GetProperty("parts");
                Assert.Equal(1, parts.GetArrayLength());
                Assert.Equal(i == 0 ? "first" : "second", parts[0].GetProperty("text").GetString());
            }
            return JsonResponse("""{"embeddings":[{"values":[1,0]},{"values":[0,1]}]}""");
        }));
        var response = await CreateProvider(http).EmbedBatchAsync(new ProviderBatchEmbeddingRequest
        {
            ModelApiName = "gemini-embedding-2", Inputs = ["first", "second"],
        }, CancellationToken.None);
        Assert.Equal(2, response.Embeddings.Count);
        Assert.Equal(new float[] { 1, 0 }, response.Embeddings[0].ToArray());
        Assert.Equal(new float[] { 0, 1 }, response.Embeddings[1].ToArray());
    }

    [Theory]
    [InlineData("""{"embedding":{"values":[1,0]},"usageMetadata":{"promptTokenCount":17}}""", 17)]
    [InlineData("""{"embedding":{"values":[1,0]}}""", 0)]
    [InlineData("""{"embedding":{"values":[1,0]},"usageMetadata":{}}""", 0)]
    public async Task Embedding_ReportsAvailableInputUsage(string responseJson, int expectedInputTokens)
    {
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(JsonResponse(responseJson))));
        var response = await CreateProvider(http).EmbedAsync(new ProviderEmbeddingRequest
        {
            ModelApiName = "gemini-embedding-2", Input = "text",
        }, CancellationToken.None);
        Assert.Equal(new float[] { 1, 0 }, response.Embedding.ToArray());
        Assert.Equal(expectedInputTokens, response.Usage.InputTokens);
        Assert.Equal(0, response.Usage.OutputTokens);
    }

    [Theory]
    [InlineData("""{"embeddings":[{"values":[1],"usageMetadata":{"promptTokenCount":10}},{"values":[2],"usageMetadata":{"promptTokenCount":20}}],"usageMetadata":{"promptTokenCount":40}}""", 40)]
    [InlineData("""{"embeddings":[{"values":[1],"usageMetadata":{"promptTokenCount":10}},{"values":[2],"usageMetadata":{"promptTokenCount":20}}],"usageMetadata":{"promptTokenCount":0}}""", 0)]
    [InlineData("""{"embeddings":[{"values":[1],"usageMetadata":{"promptTokenCount":10}},{"values":[2],"usageMetadata":{"promptTokenCount":20}}],"usageMetadata":{}}""", 30)]
    [InlineData("""{"embeddings":[{"values":[1],"usageMetadata":{"promptTokenCount":10}},{"values":[2]}]}""", 10)]
    [InlineData("""{"embeddings":[{"values":[1]},{"values":[2]}]}""", 0)]
    public async Task BatchEmbedding_PrefersTotalUsageAndOtherwiseSumsAvailableItems(string responseJson, int expectedInputTokens)
    {
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(JsonResponse(responseJson))));
        var response = await CreateProvider(http).EmbedBatchAsync(new ProviderBatchEmbeddingRequest
        {
            ModelApiName = "gemini-embedding-2", Inputs = ["first", "second"],
        }, CancellationToken.None);
        Assert.Equal(2, response.Embeddings.Count);
        Assert.Equal(new float[] { 1 }, response.Embeddings[0].ToArray());
        Assert.Equal(new float[] { 2 }, response.Embeddings[1].ToArray());
        Assert.Equal(expectedInputTokens, response.Usage.InputTokens);
        Assert.Equal(0, response.Usage.OutputTokens);
    }

    private static ProviderRequest Request(string model) => new()
    {
        ModelApiName = model,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    };

    private static GoogleProvider CreateProvider(HttpClient http) => new(new ProviderRequestContext
    {
        HttpClient = http,
        BaseUrl = "https://generativelanguage.googleapis.com",
        ApiKey = "test",
    });

    private static HttpResponseMessage JsonResponse(string json) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
    };
}
