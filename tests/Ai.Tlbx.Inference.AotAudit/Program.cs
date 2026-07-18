using System.Text.Json;
using System.Text.Json.Serialization;
using Ai.Tlbx.Inference;
using Ai.Tlbx.Inference.Configuration;

var options = new AiInferenceOptions();
options.AddOpenAi("sk-test");
options.AddAnthropic("sk-ant-test");
options.AddGoogle("AIza-test");
options.AddXai("xai-test");

using var httpClient = new HttpClient();
var client = new AiInferenceClient(httpClient, options);

var toolParameters = JsonDocument.Parse("""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""").RootElement.Clone();
var requestSchema = """{"type":"object","properties":{"city":{"type":"string"},"temperature":{"type":"number"}},"required":["city","temperature"]}""";

var completionRequest = new CompletionRequest
{
    Model = AiModel.Gpt52,
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Hello" }],
    SystemMessage = "You are a helpful assistant.",
    Temperature = 0.7,
    MaxTokens = 1000,
    ThinkingBudget = 5000,
    EnableCache = true,
    JsonSchema = requestSchema,
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
    Model = ImageGenerationModel.GptImage2,
    Prompt = "A sunset",
};

var tools = new List<ToolDefinition>
{
    new()
    {
        Name = "get_weather",
        Description = "Get weather for a city",
        ParametersSchema = toolParameters,
    },
};

Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor = call => Task.FromResult(new ToolCallResult
{
    ToolCallId = call.Id,
    Result = """{"temp":20}""",
});

// Safe AOT path.
_ = client.CompleteAsync(completionRequest);
_ = client.CompleteAsync(completionRequest, AotAuditJsonContext.Default.WeatherInfo);
_ = client.StreamAsync(completionRequest);
_ = client.CompleteWithToolsAsync(completionRequest, tools, toolExecutor);
_ = client.CompleteWithToolsAsync(completionRequest, tools, toolExecutor, AotAuditJsonContext.Default.WeatherInfo);
_ = client.StreamWithToolsAsync(completionRequest, tools, toolExecutor);
_ = client.EmbedAsync(embeddingRequest);
_ = client.EmbedBatchAsync(batchEmbeddingRequest);
_ = client.GenerateImageAsync(imageRequest);

Console.WriteLine("AOT audit app compiled.");

public sealed record WeatherInfo
{
    public required string City { get; init; }
    public required double Temperature { get; init; }
}

[JsonSerializable(typeof(WeatherInfo))]
internal partial class AotAuditJsonContext : JsonSerializerContext
{
}
