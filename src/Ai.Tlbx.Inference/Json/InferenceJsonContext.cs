using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ai.Tlbx.Inference.Json;

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(CompletionRequest))]
[JsonSerializable(typeof(ChatMessage))]
[JsonSerializable(typeof(ChatMessage[]))]
[JsonSerializable(typeof(TokenUsage))]
[JsonSerializable(typeof(ToolDefinition))]
[JsonSerializable(typeof(ToolCallRequest))]
[JsonSerializable(typeof(ToolCallResult))]
[JsonSerializable(typeof(DocumentAttachment))]
[JsonSerializable(typeof(EmbeddingRequest))]
[JsonSerializable(typeof(EmbeddingResponse))]
[JsonSerializable(typeof(BatchEmbeddingRequest))]
[JsonSerializable(typeof(BatchEmbeddingResponse))]
[JsonSerializable(typeof(ImageGenerationRequest))]
[JsonSerializable(typeof(CompletionResponse<string>))]
[JsonSerializable(typeof(JsonElement))]
internal partial class InferenceJsonContext : JsonSerializerContext
{
}
