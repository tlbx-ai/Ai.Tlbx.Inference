using System.Text.Json.Serialization;

namespace Ai.Tlbx.Inference.Json;

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(CompletionRequest))]
[JsonSerializable(typeof(ChatMessage))]
[JsonSerializable(typeof(TokenUsage))]
[JsonSerializable(typeof(TokenCostRates))]
[JsonSerializable(typeof(TokenCostEstimate))]
[JsonSerializable(typeof(ToolDefinition))]
[JsonSerializable(typeof(ToolCallRequest))]
[JsonSerializable(typeof(ToolCallResult))]
[JsonSerializable(typeof(EmbeddingRequest))]
[JsonSerializable(typeof(EmbeddingResponse))]
[JsonSerializable(typeof(BatchEmbeddingRequest))]
[JsonSerializable(typeof(BatchEmbeddingResponse))]
[JsonSerializable(typeof(ImageGenerationRequest))]
[JsonSerializable(typeof(CompletionResponse<string>))]
[JsonSerializable(typeof(ToolExecutionResponse<string>))]
[JsonSerializable(typeof(OpenAiFileUploadResponseDto))]
[JsonSerializable(typeof(OpenAiEmbeddingResponseDto))]
[JsonSerializable(typeof(GoogleEmbeddingResponseDto))]
[JsonSerializable(typeof(GoogleBatchEmbeddingResponseDto))]
[JsonSerializable(typeof(GoogleImageGenerationResponseDto))]
internal partial class InferenceJsonContext : JsonSerializerContext
{
}
