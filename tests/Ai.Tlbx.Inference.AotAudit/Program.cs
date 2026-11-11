using System.Text.Json;
using System.Text.Json.Serialization;
using Ai.Tlbx.Inference;
using Ai.Tlbx.Inference.Configuration;

var options = new AiInferenceOptions();
options.AddOpenAi("sk-test");
options.AddAnthropic("sk-ant-test");
options.AddGoogle("AIza-test");
options.AddXai("xai-test");
options.AddOpenRouter("router-test");
options.AddMistral("mistral-test");

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
    ToolIterationObserver = new AuditObserver(),
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

var imageEditRequest = new ImageEditRequest
{
    Model = ImageGenerationModel.GptImage2,
    Prompt = "Make the sunset purple",
    Images =
    [
        new ImageEditInput
        {
            Content = new byte[] { 0x89, 0x50, 0x4E, 0x47 },
            FileName = "sunset.png",
            MimeType = "image/png",
        },
    ],
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
_ = client.EditImageAsync(imageEditRequest);
_ = client.TranscribeAudioAsync(new AudioTranscriptionRequest { Audio = [1, 2], Language = "de" });
_ = client.GenerateSpeechAsync(new SpeechGenerationRequest { Text = "Hallo", Voice = "voice_481" });
_ = client.StreamSpeechAsync(new SpeechGenerationRequest { Host = AudioHost.Mistral, Text = "Hallo", Voice = "voice_481" });

using var hostedHttp = new HttpClient(new HostedAuditHandler());
var hosted = new AiInferenceClient(hostedHttp, options);
var session = await hosted.CreateHostedAgentSessionAsync(new HostedAgentOptions { Model = "gpt-6-astra" });
_ = await hosted.GetHostedAgentSessionAsync(session.Id);
_ = await hosted.ListHostedAgentItemsAsync(session.Id);
_ = await hosted.ListHostedAgentTurnsAsync(session.Id);
await foreach (var item in hosted.RunHostedAgentTurnAsync(session.Id, "AOT mock"))
    if (item.IsRootTurn != true) throw new InvalidOperationException("Unexpected mock turn.");
await hosted.CancelHostedAgentTurnAsync(session.Id);
await hosted.DeleteHostedAgentSessionAsync(session.Id);
Console.WriteLine("AOT audit app compiled; hosted-agent mock lifecycle passed.");

public sealed record WeatherInfo
{
    public required string City { get; init; }
    public required double Temperature { get; init; }
}

internal sealed class AuditObserver : IToolIterationObserver
{
    public ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken) => ValueTask.CompletedTask;
    public ValueTask OnFinishedAsync(ToolIterationResult result) => ValueTask.CompletedTask;
}

[JsonSerializable(typeof(WeatherInfo))]
internal partial class AotAuditJsonContext : JsonSerializerContext
{
}

internal sealed class HostedAuditHandler : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var text = request.Method == HttpMethod.Get && request.RequestUri!.AbsolutePath.EndsWith("/events")
            ? "data: {\"type\":\"agent.session.turn.completed\",\"turn\":{\"id\":\"root\",\"subagent_id\":null}}\n\n"
            : request.RequestUri!.AbsolutePath.EndsWith("/items") || request.RequestUri.AbsolutePath.EndsWith("/turns")
                ? "{\"data\":[],\"has_more\":false}"
                : "{\"id\":\"sess_mock\",\"status\":\"idle\"}";
        return Task.FromResult(new HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new StringContent(text) });
    }
}
