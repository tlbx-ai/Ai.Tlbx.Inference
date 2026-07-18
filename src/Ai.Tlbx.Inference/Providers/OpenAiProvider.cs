using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class OpenAiProvider : OpenAiCompatibleProvider
{
    public OpenAiProvider(ProviderRequestContext context) : base(context)
    {
    }

    protected override string MapReasoningEffort(int thinkingBudget) => thinkingBudget switch
    {
        < 5000 => "low",
        <= 20000 => "medium",
        _ => "high",
    };

    public override async Task<byte[]> GenerateImageAsync(ProviderImageRequest request, CancellationToken ct)
    {
        var body = new JsonObject
        {
            ["model"] = request.ModelApiName,
            ["prompt"] = request.Prompt,
            ["n"] = 1,
            ["output_format"] = "png"
        };

        if (!string.IsNullOrWhiteSpace(request.Size))
        {
            body["size"] = request.Size;
        }

        if (!string.IsNullOrWhiteSpace(request.Quality))
        {
            body["quality"] = request.Quality;
        }

        var jsonBytes = SerializeToUtf8Bytes(body);
        _context.Log?.Invoke(InferenceLogLevel.Debug, $"Request to {_context.BaseUrl}/v1/images/generations");

        using var response = await _context.SendAsync(
            () => CreateHttpRequest(jsonBytes, "/v1/images/generations"),
            ct).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new HttpRequestException(
                $"OpenAI image generation failed with status {response.StatusCode}: {errorBody}",
                null,
                response.StatusCode);
        }

        using var responseStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using var document = await JsonDocument.ParseAsync(responseStream, cancellationToken: ct).ConfigureAwait(false);

        if (!document.RootElement.TryGetProperty("data", out var data)
            || data.ValueKind != JsonValueKind.Array
            || data.GetArrayLength() == 0
            || !data[0].TryGetProperty("b64_json", out var base64Element)
            || base64Element.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(base64Element.GetString()))
        {
            throw new InvalidOperationException("OpenAI image generation response did not include image data.");
        }

        return Convert.FromBase64String(base64Element.GetString()!);
    }
}
