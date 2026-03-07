using System.Globalization;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class GoogleTokenProvider
{
    private const string Scope = "https://www.googleapis.com/auth/cloud-platform";
    private static readonly TimeSpan TokenLifetime = TimeSpan.FromMinutes(55);
    private static readonly TimeSpan TokenRefreshSkew = TimeSpan.FromMinutes(5);

    private readonly HttpClient _httpClient;
    private readonly string _clientEmail;
    private readonly string _tokenUri;
    private readonly RSA _privateKey;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private string? _accessToken;
    private DateTimeOffset _expiresAtUtc;

    public GoogleTokenProvider(HttpClient httpClient, string serviceAccountJson)
    {
        _httpClient = httpClient;

        using var document = JsonDocument.Parse(serviceAccountJson);
        var root = document.RootElement;

        _clientEmail = root.GetProperty("client_email").GetString()
            ?? throw new InvalidOperationException("Google service account json is missing client_email.");
        _tokenUri = root.TryGetProperty("token_uri", out var tokenUriElement)
            ? tokenUriElement.GetString() ?? "https://oauth2.googleapis.com/token"
            : "https://oauth2.googleapis.com/token";

        var privateKeyPem = root.GetProperty("private_key").GetString()
            ?? throw new InvalidOperationException("Google service account json is missing private_key.");

        _privateKey = RSA.Create();
        _privateKey.ImportFromPem(privateKeyPem);
    }

    public async Task<string> GetAccessTokenAsync(CancellationToken ct)
    {
        if (HasValidCachedToken())
        {
            return _accessToken!;
        }

        await _lock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (HasValidCachedToken())
            {
                return _accessToken!;
            }

            var now = DateTimeOffset.UtcNow;
            var assertion = CreateJwtAssertion(now);
            using var content = new FormUrlEncodedContent(
            [
                new KeyValuePair<string, string>("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                new KeyValuePair<string, string>("assertion", assertion),
            ]);

            using var request = new HttpRequestMessage(HttpMethod.Post, _tokenUri)
            {
                Content = content,
            };
            request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            using var response = await _httpClient.SendAsync(request, ct).ConfigureAwait(false);
            var responseBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException(
                    $"Google OAuth token request failed with status {response.StatusCode}: {responseBody}",
                    null,
                    response.StatusCode);
            }

            using var responseDocument = JsonDocument.Parse(responseBody);
            var responseRoot = responseDocument.RootElement;

            _accessToken = responseRoot.GetProperty("access_token").GetString()
                ?? throw new InvalidOperationException("Google OAuth token response did not include access_token.");

            var expiresInSeconds = responseRoot.TryGetProperty("expires_in", out var expiresInElement)
                ? expiresInElement.GetInt32()
                : (int)TokenLifetime.TotalSeconds;
            _expiresAtUtc = now.AddSeconds(expiresInSeconds);

            return _accessToken;
        }
        finally
        {
            _lock.Release();
        }
    }

    private bool HasValidCachedToken()
        => _accessToken is not null && DateTimeOffset.UtcNow < _expiresAtUtc.Subtract(TokenRefreshSkew);

    private string CreateJwtAssertion(DateTimeOffset now)
    {
        var headerJson = """{"alg":"RS256","typ":"JWT"}""";
        var issuedAt = now.ToUnixTimeSeconds();
        var expiresAt = now.Add(TokenLifetime).ToUnixTimeSeconds();
        var payloadJson = string.Create(
            CultureInfo.InvariantCulture,
            $$"""{"iss":"{{_clientEmail}}","scope":"{{Scope}}","aud":"{{_tokenUri}}","iat":{{issuedAt}},"exp":{{expiresAt}}}""");

        var encodedHeader = Base64UrlEncode(Encoding.UTF8.GetBytes(headerJson));
        var encodedPayload = Base64UrlEncode(Encoding.UTF8.GetBytes(payloadJson));
        var signingInput = $"{encodedHeader}.{encodedPayload}";
        var signature = _privateKey.SignData(
            Encoding.ASCII.GetBytes(signingInput),
            HashAlgorithmName.SHA256,
            RSASignaturePadding.Pkcs1);

        return $"{signingInput}.{Base64UrlEncode(signature)}";
    }

    private static string Base64UrlEncode(ReadOnlySpan<byte> bytes)
    {
        var base64 = Convert.ToBase64String(bytes);
        return base64.TrimEnd('=').Replace('+', '-').Replace('/', '_');
    }
}
