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

using var nativeHttp = new HttpClient(new NativeContextAuditHandler());
var native = new AiInferenceClient(nativeHttp, options);
var nativeRequest = new CompletionRequest
{
    Model = AiModel.Gpt61Terra,
    SystemMessage = "rules",
    Messages = [new ChatMessage { Role = ChatRole.User, Content = "Keep the worksheet intact." }],
    ConversationObserver = new NativeJournalAuditObserver(),
    ToolIterationObserver = new NativeCountAuditObserver(),
};
if (await native.CountInputTokensAsync(nativeRequest, tools) != 500) throw new InvalidOperationException("Native count failed.");
var nativeCompact = await native.CompactAsync(new CompactionRequest { Model = nativeRequest.Model, SystemMessage = nativeRequest.SystemMessage, Messages = nativeRequest.Messages });
if (nativeCompact.Messages.Count != 3 || nativeCompact.Messages.Any(item => item.Role != ChatRole.Provider)) throw new InvalidOperationException("Native compact window was lost.");
var nativeResult = await native.CompleteWithToolsAsync(nativeRequest with { Compaction = new CompactionPolicy { ThresholdTokens = 200 } }, tools, toolExecutor);
if (nativeResult.Content != "native AOT passed" || nativeResult.CompactionState?.Messages.Count != 3 || nativeResult.Messages.Count != 1)
    throw new InvalidOperationException("Native checkpoint or append-only journal failed.");
Console.WriteLine("Native compaction/counting, complete-window replay and observer mocks passed.");

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

internal sealed class NativeJournalAuditObserver : IConversationObserver
{
    public ValueTask OnMessageAsync(ChatMessage message, CancellationToken cancellationToken)
    {
        if (message.Role != ChatRole.Assistant || message.Content != "native AOT passed") throw new InvalidOperationException("Native journal callback failed.");
        return ValueTask.CompletedTask;
    }
}

internal sealed class NativeCountAuditObserver : IToolIterationObserver
{
    public ValueTask OnStartingAsync(ToolIterationRequest request, CancellationToken cancellationToken)
    {
        if (request.InputTokens != 100) throw new InvalidOperationException("Compacted exact token count was lost.");
        return ValueTask.CompletedTask;
    }
    public ValueTask OnFinishedAsync(ToolIterationResult result) => ValueTask.CompletedTask;
}

internal sealed class NativeContextAuditHandler : HttpMessageHandler
{
    private const string Window = """[{"role":"developer","content":"rules"},{"role":"user","content":[{"type":"input_file","filename":"worksheet.pdf","file_data":"data:application/pdf;base64,JVBERg=="}]},{"type":"compaction","id":"cmp_aot","encrypted_content":"opaque-AOT"}]""";
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        using var document = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
        var input = document.RootElement.GetProperty("input");
        var compacted = input.EnumerateArray().Any(item => item.TryGetProperty("type", out var type) && type.GetString() == "compaction");
        var path = request.RequestUri!.AbsolutePath;
        string response;
        if (path.EndsWith("/input_tokens", StringComparison.Ordinal)) response = compacted ? "{\"input_tokens\":100}" : "{\"input_tokens\":500}";
        else if (path.EndsWith("/compact", StringComparison.Ordinal)) response = "{\"id\":\"compact_aot\",\"output\":" + Window + ",\"usage\":{\"input_tokens\":500,\"output_tokens\":100}}";
        else
        {
            if (!compacted || input.GetArrayLength() != 3 || input[1].GetProperty("content")[0].GetProperty("file_data").GetString() != "data:application/pdf;base64,JVBERg==")
                throw new InvalidOperationException("Native AOT request mutated the complete window.");
            response = """{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"native AOT passed"}]}],"usage":{"input_tokens":100,"output_tokens":4}}""";
        }
        return new HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new StringContent(response) };
    }
}
