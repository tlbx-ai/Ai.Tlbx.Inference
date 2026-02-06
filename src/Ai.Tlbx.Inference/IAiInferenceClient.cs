using System.Diagnostics.CodeAnalysis;
using System.Text.Json.Serialization.Metadata;

namespace Ai.Tlbx.Inference;

public interface IAiInferenceClient
{
    Task<CompletionResponse<string>> CompleteAsync(CompletionRequest request, CancellationToken ct = default);

    [RequiresUnreferencedCode("Use the overload accepting JsonTypeInfo<T> for AOT compatibility.")]
    [RequiresDynamicCode("Use the overload accepting JsonTypeInfo<T> for AOT compatibility.")]
    Task<CompletionResponse<T>> CompleteAsync<T>(CompletionRequest request, CancellationToken ct = default);

    Task<CompletionResponse<T>> CompleteAsync<T>(CompletionRequest request, JsonTypeInfo<T> jsonTypeInfo, CancellationToken ct = default);

    IAsyncEnumerable<string> StreamAsync(CompletionRequest request, CancellationToken ct = default);

    [RequiresUnreferencedCode("Use the overload accepting JsonTypeInfo<T> for AOT compatibility.")]
    [RequiresDynamicCode("Use the overload accepting JsonTypeInfo<T> for AOT compatibility.")]
    Task<ToolExecutionResponse<T>> CompleteWithToolsAsync<T>(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        int maxIterations = 20,
        CancellationToken ct = default);

    Task<ToolExecutionResponse<T>> CompleteWithToolsAsync<T>(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        JsonTypeInfo<T> jsonTypeInfo,
        int maxIterations = 20,
        CancellationToken ct = default);

    IAsyncEnumerable<ToolStreamEvent> StreamWithToolsAsync(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
        int maxIterations = 20,
        CancellationToken ct = default);

    Task<EmbeddingResponse> EmbedAsync(EmbeddingRequest request, CancellationToken ct = default);

    Task<BatchEmbeddingResponse> EmbedBatchAsync(BatchEmbeddingRequest request, CancellationToken ct = default);

    Task<byte[]> GenerateImageAsync(ImageGenerationRequest request, CancellationToken ct = default);
}
