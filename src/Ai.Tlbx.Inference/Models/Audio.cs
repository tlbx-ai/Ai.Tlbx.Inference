namespace Ai.Tlbx.Inference;

/// <summary>The service that authenticates and routes audio requests.</summary>
public enum AudioHost { OpenRouter, Mistral }

/// <summary>Pinned Voxtral 3 model IDs, without a routing prefix.</summary>
public static class VoxtralModels
{
    public const string MiniTranscribe3 = "voxtral-mini-transcribe-3-0";
    public const string SmallTranscribe3 = "voxtral-small-transcribe-3-0";
    public const string MiniTts3 = "voxtral-mini-tts-3-0";
}

public sealed record AudioTranscriptionRequest
{
    public AudioHost Host { get; init; } = AudioHost.OpenRouter;
    public string Model { get; init; } = VoxtralModels.MiniTranscribe3;
    public required byte[] Audio { get; init; }
    public string Format { get; init; } = "wav";
    public string? Language { get; init; }
}

public sealed record AudioTranscriptionResponse
{
    public required string Text { get; init; }
    public required string Model { get; init; }
    public string? GenerationId { get; init; }
    public string? RawUsageJson { get; init; }
}

public sealed record SpeechGenerationRequest
{
    public AudioHost Host { get; init; } = AudioHost.OpenRouter;
    public string Model { get; init; } = VoxtralModels.MiniTts3;
    public required string Text { get; init; }
    public required string Voice { get; init; }
    /// <summary>OpenRouter supports mp3 and pcm. Direct Mistral additionally supports wav, flac and opus.
    /// Raw PCM encoding and sample rate are provider-specific; do not assume PCM16.</summary>
    public string Format { get; init; } = "mp3";
}
