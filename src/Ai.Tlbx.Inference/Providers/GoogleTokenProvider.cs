using Google.Apis.Auth.OAuth2;

namespace Ai.Tlbx.Inference.Providers;

internal sealed class GoogleTokenProvider
{
    private static readonly string[] _scopes = ["https://www.googleapis.com/auth/cloud-platform"];

    private readonly ServiceAccountCredential _credential;

    public GoogleTokenProvider(string serviceAccountJson)
    {
        var googleCredential = CredentialFactory.FromJson<ServiceAccountCredential>(serviceAccountJson)
            .ToGoogleCredential()
            .CreateScoped(_scopes);

        _credential = (ServiceAccountCredential)googleCredential.UnderlyingCredential;
    }

    public async Task<string> GetAccessTokenAsync(CancellationToken ct)
    {
        var token = await _credential.GetAccessTokenForRequestAsync(cancellationToken: ct).ConfigureAwait(false);
        return token;
    }
}
