using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Ai.Tlbx.Inference;
using Ai.Tlbx.Inference.Configuration;

var options = new AiInferenceOptions();
options.AddOpenAi("sk-test");
options.AddAnthropic("sk-ant-test");
options.AddGoogle("AIza-test");
options.AddXai("xai-test");

using var httpClient = new HttpClient();
var client = new AiInferenceClient(httpClient, options);

var completionRequest = new CompletionRequest
{
    Model = AiModel.Gpt52,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    SystemMessage = "You are a helpful assistant.",
    Temperature = 0.7,
    MaxTokens = 1000,
    ThinkingBudget = 5000,
    EnableCache = true,
    JsonSchema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""",
};

var embeddingRequest = new EmbeddingRequest
{
    Model = EmbeddingModel.TextEmbedding3Large,
    Input = "test input",
};

var batchEmbeddingRequest = new BatchEmbeddingRequest
{
    Model = EmbeddingModel.GeminiEmbedding001,
    Inputs = ["one", "two", "three"],
};

var imageRequest = new ImageGenerationRequest
{
    Prompt = "A sunset",
};

var tools = new List<ToolDefinition>
{
    new()
    {
        Name = "get_weather",
        Description = "Get weather for a city",
        ParametersSchema = JsonDocument.Parse("""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""").RootElement,
    },
};

Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor = call => Task.FromResult(new ToolCallResult
{
    ToolCallId = call.Id,
    Result = """{"temp":20}""",
});

// AOT-safe overloads with JsonTypeInfo<T>
_ = client.CompleteAsync(completionRequest, AotTestJsonContext.Default.WeatherInfo);
_ = client.CompleteWithToolsAsync(completionRequest, tools, toolExecutor, AotTestJsonContext.Default.WeatherInfo);

// Non-generic overloads (always AOT-safe)
_ = client.CompleteAsync(completionRequest);
_ = client.StreamAsync(completionRequest);
_ = client.StreamWithToolsAsync(completionRequest, tools, toolExecutor);
_ = client.EmbedAsync(embeddingRequest);
_ = client.EmbedBatchAsync(batchEmbeddingRequest);
_ = client.GenerateImageAsync(imageRequest);

Console.WriteLine("AOT testbed compiled successfully.");

public sealed record WeatherInfo
{
    public required string City { get; init; }
    public required double Temperature { get; init; }
}

[JsonSerializable(typeof(WeatherInfo))]
internal partial class AotTestJsonContext : JsonSerializerContext
{
}
