using System.Net;
using Ai.Tlbx.Inference.Configuration;

namespace Ai.Tlbx.Inference.IntegrationTests;

internal static class IntegrationScenarioHelper
{
    public static IEnumerable<object[]> GetModelsForProvider(ProviderType providerType)
        => AiModelCatalog
            .GetByProvider(providerType)
            .Select(model => new object[] { model.Model });

    public static IEnumerable<object[]> GetAttachmentModelsForProvider(ProviderType providerType)
        => AiModelCatalog
            .GetByProvider(providerType)
            .Where(model => model.Capabilities.SupportsDocumentAttachments)
            .Select(model => new object[] { model.Model });

    public static string RequireEnv(string variableName)
    {
        var value = Environment.GetEnvironmentVariable(variableName);
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException($"Expected environment variable {variableName} to be set.");
        }

        return value;
    }

    public static TestClient CreateClient(ProviderType providerType)
    {
        var options = new AiInferenceOptions();
        switch (providerType)
        {
            case ProviderType.OpenAi:
                options.AddOpenAi(RequireEnv("OPENAI_API_KEY"));
                break;
            case ProviderType.Anthropic:
                options.AddAnthropic(RequireEnv("ANTHROPIC_API_KEY"));
                break;
            case ProviderType.Google:
                options.AddGoogle(RequireEnv("GOOGLE_API_KEY"));
                break;
            case ProviderType.Xai:
                options.AddXai(RequireEnv("XAI_API_KEY"));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(providerType), providerType, null);
        }

        var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(3),
        };

        return new TestClient(httpClient, new AiInferenceClient(httpClient, options));
    }

    public static async Task<CompletionResponse<string>> ExecuteSmokeRequestAsync(AiInferenceClient client, AiModel model, CancellationToken ct = default)
    {
        var manifestEntry = LiveTestManifest.Entries.Single(entry => entry.Model == model);

        var response = await client.CompleteAsync(CompletionRequestProfiles.CreateSmoke(model) with
        {
            MaxTokens = manifestEntry.DefaultSmokeMaxTokens,
        }, ct);

        if (!string.IsNullOrWhiteSpace(response.Content))
        {
            return response;
        }

        return await client.CompleteAsync(CompletionRequestProfiles.CreateSmokeRetry(model) with
        {
            MaxTokens = manifestEntry.RetrySmokeMaxTokens,
        }, ct);
    }

    public static async Task<IReadOnlyList<string>> ExecuteStreamingSmokeAsync(AiInferenceClient client, AiModel model, CancellationToken ct = default)
    {
        var initialRequest = CompletionRequestProfiles.CreateStreamingSmoke(
            model,
            """
            Write exactly twelve lines in plain text.
            Each line must start with "Line N:" where N is 1 through 12.
            Make every line a complete sentence of at least fourteen words.
            Do not use markdown, bullets, or blank lines.
            """
        );

        var initialChunks = await CollectStreamChunksAsync(client, initialRequest, ct);
        if (initialChunks.Count > 1 && string.Concat(initialChunks).Contains("Line 12:", StringComparison.Ordinal))
        {
            return initialChunks;
        }

        var descriptor = AiModelCatalog.Get(model);
        var retryRequest = initialRequest with
        {
            MaxTokens = Math.Max(descriptor.Capabilities.StreamingSmokeMaxTokens * 2, 1536),
        };

        return await CollectStreamChunksAsync(client, retryRequest, ct);
    }

    public static async Task<CompletionResponse<string>> ExecuteAttachmentSmokeAsync(AiInferenceClient client, AiModel model, CancellationToken ct = default)
    {
        var attachment = CreatePdfAttachment();
        var request = CompletionRequestProfiles.CreateAttachmentSmoke(
            model,
            attachment,
            "Read the attached document and reply exactly with: Status=GREEN; Code=42");

        var response = await ExecuteAttachmentRequestWithRetryAsync(client, request, ct);
        if (response.Content.Contains("GREEN", StringComparison.OrdinalIgnoreCase) &&
            response.Content.Contains("42", StringComparison.OrdinalIgnoreCase))
        {
            return response;
        }

        var descriptor = AiModelCatalog.Get(model);
        return await ExecuteAttachmentRequestWithRetryAsync(client, request with
        {
            MaxTokens = Math.Max(descriptor.Capabilities.AttachmentSmokeMaxTokens * 2, 1024),
        }, ct);
    }

    public static DocumentAttachment CreatePdfAttachment()
    {
        return new DocumentAttachment
        {
            FileName = "atlas-notes.pdf",
            MimeType = "application/pdf",
            Content = BuildPdfBytes(),
        };
    }

    private static async Task<List<string>> CollectStreamChunksAsync(AiInferenceClient client, CompletionRequest request, CancellationToken ct)
    {
        var chunks = new List<string>();
        await foreach (var chunk in client.StreamAsync(request, ct))
        {
            chunks.Add(chunk);
        }

        return chunks;
    }

    private static async Task<CompletionResponse<string>> ExecuteAttachmentRequestWithRetryAsync(
        AiInferenceClient client,
        CompletionRequest request,
        CancellationToken ct)
    {
        try
        {
            return await client.CompleteAsync(request, ct);
        }
        catch (HttpRequestException ex) when (IsTransientServerError(ex))
        {
            await Task.Delay(TimeSpan.FromSeconds(2));
            return await client.CompleteAsync(request, ct);
        }
    }

    private static byte[] BuildPdfBytes()
    {
        const string pageText1 = "Project Atlas Status GREEN Code 42 Owner Ada";
        const string pageText2 = "Attachment smoke test document for provider verification.";

        var contentStream = string.Join(
            "\n",
            "BT",
            "/F1 18 Tf",
            "72 720 Td",
            $"({EscapePdfText(pageText1)}) Tj",
            "0 -28 Td",
            $"({EscapePdfText(pageText2)}) Tj",
            "ET");

        var objects = new[]
        {
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            $"<< /Length {contentStream.Length} >>\nstream\n{contentStream}\nendstream",
        };

        var builder = new System.Text.StringBuilder();
        builder.Append("%PDF-1.4\n");

        var offsets = new int[objects.Length + 1];
        for (var i = 0; i < objects.Length; i++)
        {
            offsets[i + 1] = builder.Length;
            builder.Append(i + 1).Append(" 0 obj\n");
            builder.Append(objects[i]).Append('\n');
            builder.Append("endobj\n");
        }

        var xrefOffset = builder.Length;
        builder.Append("xref\n");
        builder.Append("0 ").Append(objects.Length + 1).Append('\n');
        builder.Append("0000000000 65535 f \n");

        for (var i = 1; i <= objects.Length; i++)
        {
            builder.Append(offsets[i].ToString("D10"))
                .Append(" 00000 n \n");
        }

        builder.Append("trailer\n");
        builder.Append("<< /Size ").Append(objects.Length + 1).Append(" /Root 1 0 R >>\n");
        builder.Append("startxref\n");
        builder.Append(xrefOffset).Append('\n');
        builder.Append("%%EOF\n");

        return System.Text.Encoding.ASCII.GetBytes(builder.ToString());
    }

    private static string EscapePdfText(string text)
        => text.Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("(", "\\(", StringComparison.Ordinal)
            .Replace(")", "\\)", StringComparison.Ordinal);

    private static bool IsTransientServerError(HttpRequestException ex)
        => ex.StatusCode is HttpStatusCode.InternalServerError
            or HttpStatusCode.BadGateway
            or HttpStatusCode.ServiceUnavailable
            or HttpStatusCode.GatewayTimeout;
}

internal sealed class TestClient : IDisposable
{
    private readonly HttpClient _httpClient;

    public TestClient(HttpClient httpClient, AiInferenceClient client)
    {
        _httpClient = httpClient;
        Client = client;
    }

    public AiInferenceClient Client { get; }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
