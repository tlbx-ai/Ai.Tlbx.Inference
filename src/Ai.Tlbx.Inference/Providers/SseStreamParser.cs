using System.Runtime.CompilerServices;
using System.Text;

namespace Ai.Tlbx.Inference.Providers;

internal static class SseStreamParser
{
    internal static async IAsyncEnumerable<string> ParseAsync(
        Stream stream,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        using var reader = new StreamReader(stream, Encoding.UTF8);

        while (!reader.EndOfStream)
        {
            ct.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);

            if (line is null)
            {
                break;
            }

            if (!line.StartsWith("data: ", StringComparison.Ordinal))
            {
                continue;
            }

            var data = line["data: ".Length..];

            if (data == "[DONE]")
            {
                yield break;
            }

            yield return data;
        }
    }
}
