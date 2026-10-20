using System.Net;
using System.Text;
using System.Text.Json;
using Ai.Tlbx.Inference.Configuration;
using Ai.Tlbx.Inference.Tests.Helpers;

namespace Ai.Tlbx.Inference.Tests;

public class AudioTests
{
    [Fact]
    public async Task OpenRouterTranscriptionUsesAudioEndpointAndKeepsUsage()
    {
        using var http = new HttpClient(new MockHttpHandler(async message => {
            Assert.Equal("https://openrouter.ai/api/v1/audio/transcriptions", message.RequestUri!.AbsoluteUri);
            Assert.Equal("Bearer", message.Headers.Authorization!.Scheme);
            Assert.Equal("test-key", message.Headers.Authorization.Parameter);
            using var json = JsonDocument.Parse(await message.Content!.ReadAsStringAsync());
            Assert.Equal("mistralai/voxtral-mini-transcribe-3-0", json.RootElement.GetProperty("model").GetString());
            Assert.Equal("AQID", json.RootElement.GetProperty("input_audio").GetProperty("data").GetString());
            Assert.Equal("de", json.RootElement.GetProperty("language").GetString());
            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("{\"text\":\"Grüße\",\"usage\":{\"cost\":0.001}}") };
            response.Headers.Add("X-Generation-Id", "generation-1"); return response;
        }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenRouter("test-key"));
        var result = await client.TranscribeAudioAsync(new() { Audio = [1, 2, 3], Language = "de" });
        Assert.Equal("Grüße", result.Text); Assert.Equal("generation-1", result.GenerationId);
        Assert.Contains("cost", result.RawUsageJson);
    }

    [Theory]
    [InlineData(AudioHost.OpenRouter, "https://openrouter.ai/api/v1/audio/speech", "voice")]
    [InlineData(AudioHost.Mistral, "https://api.mistral.ai/v1/audio/speech", "voice_id")]
    public async Task SpeechRoutesCredentialsAndPreservesBinary(AudioHost host, string endpoint, string voiceField)
    {
        using var http = new HttpClient(new MockHttpHandler(async message => {
            Assert.Equal(endpoint, message.RequestUri!.AbsoluteUri);
            using var json = JsonDocument.Parse(await message.Content!.ReadAsStringAsync());
            Assert.Equal("voice_481", json.RootElement.GetProperty(voiceField).GetString());
            Assert.Equal("mp3", json.RootElement.GetProperty("response_format").GetString());
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent([0, 255, 2, 3]) };
        }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenRouter("router").AddMistral("native"));
        Assert.Equal(new byte[] { 0, 255, 2, 3 }, await client.GenerateSpeechAsync(new() { Host = host, Text = "Guten Tag", Voice = "voice_481" }));
    }

    [Fact]
    public async Task NativeTranscriptionIsMultipartAndSpeechCanReturnJsonEnvelope()
    {
        using var http = new HttpClient(new MockHttpHandler(async message => {
            if (message.RequestUri!.AbsolutePath.EndsWith("transcriptions")) {
                Assert.IsType<MultipartFormDataContent>(message.Content);
                var form = await message.Content!.ReadAsStringAsync();
                Assert.Contains("voxtral-small-transcribe-3-0", form); Assert.Contains("audio.wav", form);
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("{\"text\":\"Hallo\"}") };
            }
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("{\"audio_data\":\"AQID\"}", Encoding.UTF8, "application/json") };
        }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddMistral("native"));
        Assert.Equal("Hallo", (await client.TranscribeAudioAsync(new() { Host = AudioHost.Mistral, Audio = [1], Model = VoxtralModels.SmallTranscribe3 })).Text);
        Assert.Equal(new byte[] { 1, 2, 3 }, await client.GenerateSpeechAsync(new() { Host = AudioHost.Mistral, Text = "Hallo", Voice = "voice_481" }));
    }

    [Fact]
    public async Task FailureAndMissingCredentialsAreNotSuccessfulAudio()
    {
        using var http = new HttpClient(new MockHttpHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.TooManyRequests) { Content = new StringContent("rate limited") })));
        var missing = new AiInferenceClient(http, new AiInferenceOptions());
        await Assert.ThrowsAsync<InvalidOperationException>(() => missing.GenerateSpeechAsync(new() { Text = "Hallo", Voice = "voice_481" }));
        var client = new AiInferenceClient(http, new AiInferenceOptions().AddOpenRouter("key"));
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => client.GenerateSpeechAsync(new() { Text = "Hallo", Voice = "voice_481" }));
        Assert.Equal(HttpStatusCode.TooManyRequests, error.StatusCode);
        await Assert.ThrowsAsync<ArgumentException>(() => client.TranscribeAudioAsync(new() { Audio = [] }));
    }
}
