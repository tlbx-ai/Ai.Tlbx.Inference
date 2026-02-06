namespace Ai.Tlbx.Inference;

public interface IAiInferenceClient
{
    Task<CompletionResponse<string>> CompleteAsync(CompletionRequest request, CancellationToken ct = default);

    Task<CompletionResponse<T>> CompleteAsync<T>(CompletionRequest request, CancellationToken ct = default);

    IAsyncEnumerable<string> StreamAsync(CompletionRequest request, CancellationToken ct = default);

    Task<ToolExecutionResponse<T>> CompleteWithToolsAsync<T>(
        CompletionRequest request,
        IReadOnlyList<ToolDefinition> tools,
        Func<ToolCallRequest, Task<ToolCallResult>> toolExecutor,
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
