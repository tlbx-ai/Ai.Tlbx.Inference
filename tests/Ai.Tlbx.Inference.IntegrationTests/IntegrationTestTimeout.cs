namespace Ai.Tlbx.Inference.IntegrationTests;

internal static class IntegrationTestTimeout
{
    public static readonly TimeSpan PerTestTimeout = TimeSpan.FromMinutes(3);

    public static async Task ExecuteAsync(Func<CancellationToken, Task> action)
    {
        using var cts = new CancellationTokenSource(PerTestTimeout);
        await action(cts.Token);
    }
}
