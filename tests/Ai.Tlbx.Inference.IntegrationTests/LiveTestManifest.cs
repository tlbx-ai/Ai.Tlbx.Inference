namespace Ai.Tlbx.Inference.IntegrationTests;

internal static class LiveTestManifest
{
    // Separate opt-in: never included merely because a provider API key is configured.
    public const string NativeCompactionAcceptanceOptIn = "AI_TLBX_RUN_NATIVE_COMPACTION_ACCEPTANCE";
    public const decimal NativeCompactionAcceptanceBudgetUsd = .50m;

    public static IReadOnlyList<LiveTestManifestEntry> Entries { get; } =
        AiModelCatalog.All
            .Select(model => new LiveTestManifestEntry
            {
                Model = model.Model,
                Provider = model.Provider,
                ApiName = model.ApiName,
                EndpointFamily = model.PreferredEndpoint,
                RequiredEnvironmentVariable = model.Provider switch
                {
                    ProviderType.OpenAi => "OPENAI_API_KEY",
                    ProviderType.Anthropic => "ANTHROPIC_API_KEY",
                    ProviderType.Google => "GOOGLE_API_KEY",
                    ProviderType.Xai => "XAI_API_KEY",
                    _ => throw new ArgumentOutOfRangeException()
                },
                DefaultSmokeMaxTokens = model.Capabilities.DefaultSmokeMaxTokens,
                RetrySmokeMaxTokens = model.Capabilities.RetrySmokeMaxTokens,
                RetryOnEmptyResponse = true,
            })
            .ToArray();
}

internal sealed record LiveTestManifestEntry
{
    public required AiModel Model { get; init; }
    public required ProviderType Provider { get; init; }
    public required string ApiName { get; init; }
    public required ModelEndpointFamily EndpointFamily { get; init; }
    public required string RequiredEnvironmentVariable { get; init; }
    public required int DefaultSmokeMaxTokens { get; init; }
    public required int RetrySmokeMaxTokens { get; init; }
    public required bool RetryOnEmptyResponse { get; init; }
}
