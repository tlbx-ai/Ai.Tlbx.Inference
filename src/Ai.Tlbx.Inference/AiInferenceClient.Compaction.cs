using Ai.Tlbx.Inference.Providers;

namespace Ai.Tlbx.Inference;

public sealed partial class AiInferenceClient
{
    public Task<int> CountInputTokensAsync(CompletionRequest request, IReadOnlyList<ToolDefinition>? tools = null, CancellationToken ct = default)
    {
        var provider = GetNativeContextProvider(request.Model, false);
        var active = ApplyCompactionState(request, request.Messages, request.CompactionState, BuildProviderRequest(request, tools: tools));
        return provider.CountInputTokensAsync(active, ct);
    }

    public Task<CompactionResponse> CompactAsync(CompactionRequest request, CancellationToken ct = default)
    {
        var provider = GetNativeContextProvider(request.Model, true);
        return provider.CompactAsync(BuildProviderRequest(new CompletionRequest
        {
            Model = request.Model, Messages = request.Messages, SystemMessage = request.SystemMessage,
            ServiceTier = request.ServiceTier, PromptCacheKey = request.PromptCacheKey
        }) with { PreferredEndpoint = ModelEndpointFamily.Responses }, ct);
    }

    private INativeContextProvider GetNativeContextProvider(AiModel model, bool compaction)
    {
        if (model.GetProvider() != ProviderType.OpenAi || !AiModelCatalog.Get(model).Capabilities.SupportsResponsesApi
            || compaction && !model.SupportsNativeCompaction())
            throw new NotSupportedException($"{model.ToDisplayName()} does not support the requested native context operation.");
        return (INativeContextProvider)GetProvider(ProviderType.OpenAi);
    }

    private static ProviderRequest ApplyCompactionState(CompletionRequest request, IReadOnlyList<ChatMessage> transcript,
        CompactionState? state, ProviderRequest prototype)
    {
        if (state is null) return prototype with { Messages = transcript };
        if (state.SourceMessageCount < 0 || state.SourceMessageCount > transcript.Count || state.Messages.Count == 0)
            throw new ArgumentException("Compaction checkpoint does not match the append-only transcript.");
        var activeMessages = state.Messages.Concat(transcript.Skip(state.SourceMessageCount));
        if (state.SystemMessage != request.SystemMessage && request.SystemMessage is { } update)
            activeMessages = activeMessages.Append(new ChatMessage { Role = ChatRole.System, Content = update });
        return prototype with
        {
            Messages = activeMessages.ToArray(),
            SystemMessage = null, PreferredEndpoint = ModelEndpointFamily.Responses
        };
    }

    private static int ContextInputCeiling(CompletionRequest request)
    {
        var ceiling = request.Model.GetContextWindow() - (request.MaxTokens ?? 4096);
        if (AiModelCostCatalog.TryGetRates(request.Model, out var rates) && rates.LongContextStartsAtTokens is int priceBoundary)
            ceiling = Math.Min(ceiling, priceBoundary - 1);
        return ceiling;
    }

    private async Task<PreparedContext> PrepareContextAsync(CompletionRequest request, IReadOnlyList<ChatMessage> transcript,
        ProviderRequest prototype, CompactionState? state, CancellationToken ct)
    {
        transcript = CloneMessages(transcript);
        var active = ApplyCompactionState(request, transcript, state, prototype);
        if (request.Compaction is not { } policy) return new(active, state, null);
        var provider = GetNativeContextProvider(request.Model, true);
        var ceiling = ContextInputCeiling(request);
        if (policy.SafetyMarginTokens < 0 || policy.ThresholdTokens is <= 0)
            throw new ArgumentOutOfRangeException(nameof(request), "Invalid compaction threshold or safety margin.");
        var threshold = Math.Min(policy.ThresholdTokens ?? int.MaxValue, ceiling - policy.SafetyMarginTokens);
        if (threshold <= 0) throw new ArgumentException("No safe input budget remains for compaction.");
        var updatedInstructions = state is not null && state.SystemMessage != request.SystemMessage;
        var inputTokens = await provider.CountInputTokensAsync(active, ct).ConfigureAwait(false);
        if (inputTokens > ceiling)
            throw new InvalidOperationException($"Input has {inputTokens} tokens, exceeding the safe {ceiling}-token budget. No higher-priced model call was sent.");
        if (inputTokens < threshold && !updatedInstructions) return new(active, state, null, inputTokens);
        // The compact operation has its own small provider prompt overhead. Leave
        // extra admission headroom rather than sending a boundary-sized compact call.
        var compactCeiling = ceiling - Math.Min(policy.SafetyMarginTokens, 4096);
        if (inputTokens > compactCeiling)
            throw new InvalidOperationException($"Input has {inputTokens} tokens, exceeding the safe native-compaction admission budget of {compactCeiling}. No higher-priced call was sent.");
        var compactRequest = new CompactionRequest
        {
            Model = request.Model, Messages = active.Messages, SystemMessage = active.SystemMessage,
            ServiceTier = request.ServiceTier, PromptCacheKey = request.PromptCacheKey, InputTokens = inputTokens
        };
        CompactionResponse? response = null;
        try
        {
            if (request.CompactionObserver is { } observer) await observer.OnStartingAsync(compactRequest, ct).ConfigureAwait(false);
            response = await provider.CompactAsync(active, ct).ConfigureAwait(false);
            var nextState = new CompactionState { Messages = response.Messages, SourceMessageCount = transcript.Count, SystemMessage = request.SystemMessage };
            var compacted = prototype with { Messages = response.Messages, SystemMessage = null, PreferredEndpoint = ModelEndpointFamily.Responses };
            var compactedTokens = -1;
            Exception? countError = null;
            try { compactedTokens = await provider.CountInputTokensAsync(compacted, ct).ConfigureAwait(false); }
            catch (Exception error) { countError = error; }
            var result = new ConversationCompaction
            {
                Response = response, State = nextState, SourceMessages = transcript.ToArray(),
                InputTokens = inputTokens, CompactedInputTokens = compactedTokens
            };
            if (request.CompactionObserver is { } completed) await completed.OnCompletedAsync(result, CancellationToken.None).ConfigureAwait(false);
            if (countError is not null) throw new InvalidOperationException("Native compaction completed, but its input budget could not be verified.", countError);
            if (compactedTokens >= threshold)
                throw new InvalidOperationException($"Native compaction retained {compactedTokens} input tokens, above the safe trigger {threshold}. No higher-priced request was sent.");
            return new(compacted, nextState, result, compactedTokens);
        }
        catch (Exception error)
        {
            if (response is null && request.CompactionObserver is { } failed)
                await failed.OnFailedAsync(error, CancellationToken.None).ConfigureAwait(false);
            throw;
        }
    }

    private sealed record PreparedContext(ProviderRequest Request, CompactionState? State, ConversationCompaction? Compaction, int? InputTokens = null);
}
