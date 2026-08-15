# Ai.Tlbx.Inference

[![NuGet](https://img.shields.io/nuget/v/Ai.Tlbx.Inference.svg)](https://www.nuget.org/packages/Ai.Tlbx.Inference)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Trim-friendly .NET AI inference client for OpenAI, Anthropic, Google, and xAI, built around a single inference facade.

## Features

- **Multi-provider** — OpenAI, Anthropic, Google (AI Studio + Vertex), xAI behind a single interface
- **Streaming** — `IAsyncEnumerable<string>` across all supported providers with integration coverage
- **Document attachments** — provider-aware document attachment support with live coverage for supported models
- **Structured output** — `CompleteAsync(..., JsonTypeInfo<T>)` for trim- and AOT-safe typed results
- **Model registry** — first-class `AiModelCatalog` descriptors with endpoint/capability metadata
- **Tool calling** — unified tool loop with streaming support, lives once in the facade
- **Embeddings** — OpenAI and Google embedding models with batch support
- **Image generation** — Google Gemini image generation via `gemini-2.5-flash-image`
- **Token metering** — `TokenUsage` on every response, with reasoning retained as a diagnostic subset of billed output tokens
- **Resilience** — Polly v8 retry and timeout handling wired into provider HTTP execution
- **Thinking budget** — universal mapping across all providers that support reasoning
- **Diagnostics** — `CompletionDiagnostics` exposes endpoint family, stop reason, and truncation hints
- **AOT/trimming-first** — manual provider JSON construction/parsing where it reduces reflection risk and wire bloat

## Design Goals

- One public facade: `IAiInferenceClient`
- Internal provider implementations
- Shared `HttpClient` per client instance
- Explicit model capability metadata instead of implicit provider heuristics
- Sparse provider payloads: omit null/default fields unless the upstream API requires them

## Supported Models

### OpenAI
| Model | Enum | Context |
|-------|------|---------|
| GPT-5.2 | `AiModel.Gpt52` | 400k |
| GPT-5.2 Pro | `AiModel.Gpt52Pro` | 400k |
| GPT-5.2 Chat | `AiModel.Gpt52Chat` | 128k |
| GPT-5.3 Chat | `AiModel.Gpt53Chat` | 128k |
| GPT-5.4 | `AiModel.Gpt54` | 1.05M |
| GPT-5.6 Luna | `AiModel.Gpt56Luna` | 1.05M |

### Anthropic
| Model | Enum | Context |
|-------|------|---------|
| Claude Opus 4.6 | `AiModel.ClaudeOpus46` | 200k |
| Claude Sonnet 4.6 | `AiModel.ClaudeSonnet46` | 200k |
| Claude Haiku 4.5 | `AiModel.ClaudeHaiku45` | 200k |

### Google
| Model | Enum | Context |
|-------|------|---------|
| Gemini 3.5 Flash | `AiModel.Gemini35Flash` | 1M |
| Gemini 3 Flash Preview | `AiModel.Gemini3FlashPreview` | 1M |
| Gemini 3.1 Pro Preview | `AiModel.Gemini31ProPreview` | 1M |
| Gemini 3.1 Flash-Lite Preview | `AiModel.Gemini31FlashLitePreview` | 1M |

### xAI
| Model | Enum | Context |
|-------|------|---------|
| Grok 4.1 Fast | `AiModel.Grok41Fast` | 2M |
| Grok 4 | `AiModel.Grok4` | 256k |

### Embeddings
| Model | Enum | Dimensions | Provider |
|-------|------|-----------|----------|
| text-embedding-3-large | `EmbeddingModel.TextEmbedding3Large` | 3072 | OpenAI |
| text-embedding-3-small | `EmbeddingModel.TextEmbedding3Small` | 1536 | OpenAI |
| gemini-embedding-001 | `EmbeddingModel.GeminiEmbedding001` | 3072 | Google |

## Installation

```
dotnet add package Ai.Tlbx.Inference
```

## Quick Start

### Model Registry

```csharp
var descriptor = AiModelCatalog.Get(AiModel.Gpt52Pro);

Console.WriteLine(descriptor.ApiName);
Console.WriteLine(descriptor.PreferredEndpoint);
Console.WriteLine(descriptor.Capabilities.SupportsResponsesApi);
```

### DI Registration

```csharp
services.AddAiInference(options =>
{
    options.AddOpenAi("sk-...");
    options.AddAnthropic("sk-ant-...");
    options.AddGoogle("AIza...");
    options.AddXai("xai-...");
});
```

### Simple Completion

```csharp
var response = await client.CompleteAsync(new CompletionRequest
{
    Model = AiModel.ClaudeOpus46,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello!" }]
});

Console.WriteLine(response.Content);
Console.WriteLine($"Tokens: {response.Usage.TotalTokens}");
Console.WriteLine(response.Diagnostics?.EndpointFamily);
Console.WriteLine(response.Diagnostics?.Note);
```

### Streaming

```csharp
await foreach (var delta in client.StreamAsync(new CompletionRequest
{
    Model = AiModel.Gpt52,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Write a haiku" }]
}))
{
    Console.Write(delta);
}
```

### Structured Output

```csharp
public sealed record WeatherInfo
{
    public required string City { get; init; }
    public required double Temperature { get; init; }
    public required string Condition { get; init; }
}

var response = await client.CompleteAsync(new CompletionRequest
{
    Model = AiModel.Gemini31FlashLitePreview,
    JsonSchema = """{"type":"object","properties":{"city":{"type":"string"},"temperature":{"type":"number"},"condition":{"type":"string"}},"required":["city","temperature","condition"]}""",
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather in Berlin?" }]
}, MyJsonContext.Default.WeatherInfo);

Console.WriteLine($"{response.Content.City}: {response.Content.Temperature}°C, {response.Content.Condition}");
```

`AiModel.Gemini35Flash` maps to Google's `gemini-3.5-flash` Gemini API model and supports text,
PDF/document attachments, streaming, structured output, and function calling through the Google
GenerateContent endpoint.

### Model Validation

```csharp
var validation = AiModelValidator.ValidateForCompletion(
    AiModel.Gpt52Pro,
    streaming: false,
    tools: false,
    structuredOutput: false);

foreach (var warning in validation.Warnings)
{
    Console.WriteLine(warning);
}
```

### Document Attachments

```csharp
var attachment = new DocumentAttachment
{
    FileName = "brief.pdf",
    MimeType = "application/pdf",
    Content = File.ReadAllBytes("brief.pdf")
};

var response = await client.CompleteAsync(new CompletionRequest
{
    Model = AiModel.Gemini31FlashLitePreview,
    Messages =
    [
        new ChatMessage
        {
            Role = ChatRole.User,
            Content = "Read the attached document and summarize it in three bullet points.",
            Attachments = [attachment]
        }
    ]
});

Console.WriteLine(response.Content);
```

### Tool Calling

```csharp
var tools = new List<ToolDefinition>
{
    new()
    {
        Name = "get_weather",
        Description = "Get current weather for a city",
        ParametersSchema = JsonDocument.Parse(
            """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""").RootElement
    }
};

var result = await client.CompleteWithToolsAsync(
    new CompletionRequest
    {
        Model = AiModel.ClaudeSonnet46,
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "What's the weather in Tokyo?" }]
    },
    tools,
    toolExecutor: async call =>
    {
        var weather = GetWeather(call.Arguments);
        return new ToolCallResult
        {
            ToolCallId = call.Id,
            Result = JsonSerializer.Serialize(weather)
        };
    });

Console.WriteLine(result.Content);
Console.WriteLine($"Tool iterations: {result.Iterations}, Total tokens: {result.Usage.TotalTokens}");
```

### Embeddings

```csharp
var embedding = await client.EmbedAsync(new EmbeddingRequest
{
    Model = EmbeddingModel.TextEmbedding3Large,
    Input = "The quick brown fox"
});

Console.WriteLine($"Dimensions: {embedding.Embedding.Length}");
```

### Image Generation

```csharp
var imageBytes = await client.GenerateImageAsync(new ImageGenerationRequest
{
    Model = ImageGenerationModel.GptImage2,
    Prompt = "A product photo of a matte black teapot on a concrete counter",
    Size = "1536x1024",
    Quality = "high"
});

await File.WriteAllBytesAsync("teapot.png", imageBytes);
```

`ImageGenerationModel` selects both the provider and the model. OpenAI image generation uses the Image API and
returns the decoded PNG bytes from `data[0].b64_json`; `Size` and `Quality` map directly to the request. Google
image generation remains available through the default `Gemini25FlashImage` model and returns the first inline
image bytes from Gemini's response.

## Configuration

### Logging

```csharp
services.AddAiInference(options =>
{
    options.AddOpenAi("sk-...");
    options.WithLogging((level, message) =>
    {
        Console.WriteLine($"[{level}] {message}");
    });
});
```

### Custom Retry Policy

```csharp
var customPipeline = new ResiliencePipelineBuilder<HttpResponseMessage>()
    .AddRetry(new RetryStrategyOptions<HttpResponseMessage>
    {
        MaxRetryAttempts = 2,
        Delay = TimeSpan.FromSeconds(5)
    })
    .Build();

services.AddAiInference(options =>
{
    options.AddOpenAi("sk-...");
    options.WithRetryPolicy(customPipeline);
});
```

### Google Vertex AI

```csharp
services.AddAiInference(options =>
{
    options.AddGoogle(
        serviceAccountJson: File.ReadAllText("service-account.json"),
        projectId: "my-project",
        location: "us-central1");
});
```

### Thinking Budget

```csharp
var response = await client.CompleteAsync(new CompletionRequest
{
    Model = AiModel.ClaudeOpus46,
    ThinkingBudget = 10000,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Solve this complex problem..." }]
});

Console.WriteLine($"Thinking tokens used: {response.Usage.ThinkingTokens}");
```

### Prompt Caching (Anthropic)

```csharp
var response = await client.CompleteAsync(new CompletionRequest
{
    Model = AiModel.ClaudeSonnet46,
    EnableCache = true,
    SystemMessage = longSystemPrompt,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Question..." }]
});

Console.WriteLine($"Cache read: {response.Usage.CacheReadTokens}, Cache write: {response.Usage.CacheWriteTokens}");
```

## AOT / Trimming

The library is AOT and trimming compatible (`IsAotCompatible`, `IsTrimmable`).

For structured output and typed tool results, use the `JsonTypeInfo<T>` overloads and provide your schema explicitly:

```csharp
[JsonSerializable(typeof(WeatherInfo))]
internal partial class MyJsonContext : JsonSerializerContext { }

var response = await client.CompleteAsync(
    new CompletionRequest
    {
        Model = AiModel.Gemini31FlashLitePreview,
        JsonSchema = """{"type":"object","properties":{"city":{"type":"string"},"temp":{"type":"number"}},"required":["city","temp"]}""",
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather in Berlin?" }]
    },
    MyJsonContext.Default.WeatherInfo);
```

The non-generic methods (`CompleteAsync`, `StreamAsync`, `EmbedAsync`, etc.) are always AOT-safe.

## Runtime Notes

- The library does not create ad hoc provider `HttpClient` instances. A single shared `HttpClient` is supplied to each `AiInferenceClient` instance and reused by all configured providers for that client.
- Provider request payloads are built manually with `JsonObject`/`JsonArray` so the library can omit unused fields and stay predictable under trimming/AOT.
- A small shared JSON policy is used for serializer-based paths, and source-generated DTOs are used for a few stable internal envelopes such as embedding and upload responses.

## Paid Integration Tests

Live provider integration tests cost money and are intended to run on demand only.

- Command: `dotnet test tests\Ai.Tlbx.Inference.IntegrationTests\Ai.Tlbx.Inference.IntegrationTests.csproj`
- Manifest: [tests/Ai.Tlbx.Inference.IntegrationTests/live-test-manifest.json](/Q:/repos/Ai.Tlbx.Inference/tests/Ai.Tlbx.Inference.IntegrationTests/live-test-manifest.json)
- Source of truth for covered models: [`AiModelCatalog`](/Q:/repos/Ai.Tlbx.Inference/src/Ai.Tlbx.Inference/Models/AiModelCatalog.cs)
- Smoke request helper: [`CompletionRequestProfiles`](/Q:/repos/Ai.Tlbx.Inference/src/Ai.Tlbx.Inference/Models/CompletionRequestProfiles.cs)

Recommended environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`

The live suite includes:

- simple prompt smoke tests
- streaming tests that verify multi-chunk output
- document attachment tests for supported models

These tests are intentionally not run automatically.

## License

[MIT](LICENSE)
