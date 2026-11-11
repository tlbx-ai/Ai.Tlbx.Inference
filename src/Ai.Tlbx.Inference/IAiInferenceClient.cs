using System.Text.Json.Serialization.Metadata;

namespace Ai.Tlbx.Inference;

public interface IAiInferenceClient
{
    Task<AudioTranscriptionResponse> TranscribeAudioAsync(AudioTranscriptionRequest request, CancellationToken ct = default);

    Task<byte[]> GenerateSpeechAsync(SpeechGenerationRequest request, CancellationToken ct = default);

    IAsyncEnumerable<byte[]> StreamSpeechAsync(SpeechGenerationRequest request, CancellationToken ct = default);

    Task<CompletionResponse<string>> CompleteAsync(CompletionRequest request, CancellationToken ct = default);

    Task<CompletionResponse<T>> CompleteAsync<T>(CompletionRequest request, JsonTypeInfo<T> jsonTypeInfo, CancellationToken ct = default);

    IAsyncEnumerable<string> StreamAsync(CompletionRequest request, CancellationToken ct = default);

    Task<ToolExecutionResponse<string>> CompleteWithToolsAsync(
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

    Task<byte[]> EditImageAsync(ImageEditRequest request, CancellationToken ct = default);

    Task<HostedAgentSession> CreateHostedAgentSessionAsync(HostedAgentOptions options, CancellationToken ct = default);

    Task<HostedAgentSession> GetHostedAgentSessionAsync(string sessionId, CancellationToken ct = default);

    Task<IReadOnlyList<HostedAgentItem>> ListHostedAgentItemsAsync(string sessionId, CancellationToken ct = default);

    Task<IReadOnlyList<HostedAgentTurn>> ListHostedAgentTurnsAsync(string sessionId, CancellationToken ct = default);

    /// <summary>Subscribe before submitting a message, then stream until the root turn ends.</summary>
    /// <remarks>Use one active reader per session. Disposing or cancelling the reader does not cancel remote work;
    /// use CancelHostedAgentTurnAsync explicitly. After a disconnected stream, retrieve saved items rather than resending.</remarks>
    IAsyncEnumerable<HostedAgentEvent> RunHostedAgentTurnAsync(string sessionId, string message, CancellationToken ct = default);

    Task CancelHostedAgentTurnAsync(string sessionId, CancellationToken ct = default);

    Task DeleteHostedAgentSessionAsync(string sessionId, CancellationToken ct = default);
}
