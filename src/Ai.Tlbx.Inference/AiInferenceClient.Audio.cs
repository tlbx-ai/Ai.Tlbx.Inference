using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ai.Tlbx.Inference.Configuration;

namespace Ai.Tlbx.Inference;

public sealed partial class AiInferenceClient
{
    private readonly HttpClient _audioHttpClient;
    private readonly AiInferenceOptions _audioOptions;

    public async Task<AudioTranscriptionResponse> TranscribeAudioAsync(AudioTranscriptionRequest request, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Audio);
        if (request.Audio.Length == 0) throw new ArgumentException("Audio must not be empty.", nameof(request));
        ArgumentException.ThrowIfNullOrWhiteSpace(request.Format);
        var model = AudioModel(request.Host, request.Model);
        using var message = AudioMessage(request.Host, "transcriptions");
        if (request.Host == AudioHost.OpenRouter)
        {
            var body = new JsonObject { ["model"] = model, ["input_audio"] = new JsonObject {
                ["data"] = Convert.ToBase64String(request.Audio), ["format"] = request.Format } };
            if (request.Language != null) body["language"] = request.Language;
            message.Content = AudioJson(body);
        }
        else
        {
            var form = new MultipartFormDataContent();
            message.Content = form;
            form.Add(new StringContent(model), "model");
            form.Add(new ByteArrayContent(request.Audio), "file", "audio." + request.Format);
            if (request.Language != null) form.Add(new StringContent(request.Language), "language");
        }
        using var response = await _audioHttpClient.SendAsync(message, ct).ConfigureAwait(false);
        await CheckAudioResponseAsync(response, ct).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false));
        return new AudioTranscriptionResponse {
            Text = doc.RootElement.GetProperty("text").GetString() ?? "", Model = model,
            GenerationId = response.Headers.TryGetValues("X-Generation-Id", out var ids) ? ids.FirstOrDefault() : null,
            RawUsageJson = doc.RootElement.TryGetProperty("usage", out var usage) ? usage.GetRawText() : null };
    }

    public async Task<byte[]> GenerateSpeechAsync(SpeechGenerationRequest request, CancellationToken ct = default)
    {
        using var output = new MemoryStream();
        await foreach (var chunk in StreamSpeechAsync(request, ct).ConfigureAwait(false))
            await output.WriteAsync(chunk, ct).ConfigureAwait(false);
        return output.ToArray();
    }

    public async IAsyncEnumerable<byte[]> StreamSpeechAsync(SpeechGenerationRequest request, [EnumeratorCancellation] CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(request.Text);
        ArgumentException.ThrowIfNullOrWhiteSpace(request.Voice);
        var formats = request.Host == AudioHost.OpenRouter ? new[] { "mp3", "pcm" } : new[] { "mp3", "pcm", "wav", "flac", "opus" };
        if (!formats.Contains(request.Format, StringComparer.Ordinal)) throw new ArgumentException("Unsupported audio output format.", nameof(request));
        using var message = AudioMessage(request.Host, "speech");
        message.Content = AudioJson(new JsonObject { ["model"] = AudioModel(request.Host, request.Model),
            ["input"] = request.Text, [request.Host == AudioHost.OpenRouter ? "voice" : "voice_id"] = request.Voice,
            ["response_format"] = request.Format });
        using var response = await _audioHttpClient.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        await CheckAudioResponseAsync(response, ct).ConfigureAwait(false);
        // Mistral's non-streaming native endpoint can return a JSON audio_data envelope.
        if (response.Content.Headers.ContentType?.MediaType == "application/json")
        {
            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false));
            yield return Convert.FromBase64String(doc.RootElement.GetProperty("audio_data").GetString()!);
            yield break;
        }
        await using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        var buffer = new byte[8192];
        int read;
        while ((read = await stream.ReadAsync(buffer, ct).ConfigureAwait(false)) != 0)
            yield return buffer.AsSpan(0, read).ToArray();
    }

    private HttpRequestMessage AudioMessage(AudioHost host, string operation)
    {
        if (!_audioOptions.AudioKeys.TryGetValue(host, out var key))
            throw new InvalidOperationException($"Configure {host} audio credentials first with Add{host}.");
        var endpoint = host switch { AudioHost.OpenRouter => "https://openrouter.ai/api/v1/audio/", AudioHost.Mistral => "https://api.mistral.ai/v1/audio/", _ => throw new ArgumentOutOfRangeException(nameof(host)) };
        var message = new HttpRequestMessage(HttpMethod.Post, endpoint + operation);
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        return message;
    }

    private static string AudioModel(AudioHost host, string model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        return host == AudioHost.OpenRouter && !model.Contains('/') ? "mistralai/" + model : model;
    }

    private static StringContent AudioJson(JsonObject body) => new(body.ToJsonString(), Encoding.UTF8, "application/json");

    private static async Task CheckAudioResponseAsync(HttpResponseMessage response, CancellationToken ct)
    {
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Audio API returned {(int)response.StatusCode}: {await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false)}", null, response.StatusCode);
    }
}
