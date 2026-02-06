# Ai.Tlbx.Inference

[![NuGet](https://img.shields.io/nuget/v/Ai.Tlbx.Inference.svg)](https://www.nuget.org/packages/Ai.Tlbx.Inference)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Universal .NET AI inference client for OpenAI, Anthropic, Google, and xAI.

## Features

- **Multi-provider** — OpenAI, Anthropic, Google (AI Studio + Vertex), xAI behind a single interface
- **Streaming** — `IAsyncEnumerable<string>` with zero-alloc delta-only design
- **Structured output** — `CompleteAsync<T>()` with automatic JSON schema inference
- **Tool calling** — unified tool loop with streaming support, lives once in the facade
- **Embeddings** — OpenAI and Google embedding models with batch support
- **Image generation** — Google Gemini image generation
- **Token metering** — `TokenUsage` on every response including cache and thinking tokens
- **Resilience** — Polly v8 retry with exponential backoff, jitter, and Retry-After respect
- **Thinking budget** — universal mapping across all providers that support reasoning

## Supported Models

### OpenAI
| Model | Enum | Context |
|-------|------|---------|
| GPT-5.2 | `AiModel.Gpt52` | 400k |
| GPT-5.2 Pro | `AiModel.Gpt52Pro` | 400k |
| GPT-5.2 Chat | `AiModel.Gpt52Chat` | 128k |
| GPT-5.3 Codex | `AiModel.Gpt53Codex` | 400k |
| GPT-5 | `AiModel.Gpt5` | 400k |
| o3 | `AiModel.O3` | 200k |
| o3 Pro | `AiModel.O3Pro` | 200k |
| o4 Mini | `AiModel.O4Mini` | 200k |

### Anthropic
| Model | Enum | Context |
|-------|------|---------|
| Claude Opus 4.6 | `AiModel.ClaudeOpus46` | 200k |
| Claude Opus 4.5 | `AiModel.ClaudeOpus45` | 200k |
| Claude Sonnet 4.5 | `AiModel.ClaudeSonnet45` | 200k |
| Claude Haiku 4.5 | `AiModel.ClaudeHaiku45` | 200k |

### Google
| Model | Enum | Context |
|-------|------|---------|
| Gemini 3 Pro Preview | `AiModel.Gemini3ProPreview` | 1M |
| Gemini 3 Flash Preview | `AiModel.Gemini3FlashPreview` | 1M |
| Gemini 2.5 Pro | `AiModel.Gemini25Pro` | 1M |
| Gemini 2.5 Flash | `AiModel.Gemini25Flash` | 1M |

### xAI
| Model | Enum | Context |
|-------|------|---------|
| Grok 4.1 Fast | `AiModel.Grok41Fast` | 2M |
| Grok 4 | `AiModel.Grok4` | 256k |
| Grok 3 | `AiModel.Grok3` | 131k |

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

var response = await client.CompleteAsync<WeatherInfo>(new CompletionRequest
{
    Model = AiModel.Gemini25Pro,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather in Berlin?" }]
});

Console.WriteLine($"{response.Content.City}: {response.Content.Temperature}°C, {response.Content.Condition}");
```

### Tool Calling

```csharp
var tools = new List<ToolDefinition>
{
    new()
    {
        Name = "get_weather",
        Description = "Get current weather for a city",
        ParametersSchema = JsonSchemaGenerator.Generate<WeatherParams>()
    }
};

var result = await client.CompleteWithToolsAsync<string>(
    new CompletionRequest
    {
        Model = AiModel.ClaudeSonnet45,
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
    Model = AiModel.ClaudeSonnet45,
    EnableCache = true,
    SystemMessage = longSystemPrompt,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Question..." }]
});

Console.WriteLine($"Cache read: {response.Usage.CacheReadTokens}, Cache write: {response.Usage.CacheWriteTokens}");
```

## AOT / Trimming

The library is AOT and trimming compatible (`IsAotCompatible`, `IsTrimmable`).

For structured output and tool calling with AOT, use the `JsonTypeInfo<T>` overloads and provide your schema explicitly:

```csharp
[JsonSerializable(typeof(WeatherInfo))]
internal partial class MyJsonContext : JsonSerializerContext { }

var response = await client.CompleteAsync(
    new CompletionRequest
    {
        Model = AiModel.Gemini25Pro,
        JsonSchema = """{"type":"object","properties":{"city":{"type":"string"},"temp":{"type":"number"}},"required":["city","temp"]}""",
        Messages = [new ChatMessage { Role = ChatRole.User, Content = "Weather in Berlin?" }]
    },
    MyJsonContext.Default.WeatherInfo);
```

The non-generic methods (`CompleteAsync`, `StreamAsync`, `EmbedAsync`, etc.) are always AOT-safe.

## License

[MIT](LICENSE)
