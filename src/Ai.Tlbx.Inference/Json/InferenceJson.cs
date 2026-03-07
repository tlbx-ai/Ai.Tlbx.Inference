using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace Ai.Tlbx.Inference.Json;

internal static class InferenceJson
{
    public static JsonSerializerOptions SerializerOptions { get; } = CreateSerializerOptions();

    public static object Deserialize(string json, JsonTypeInfo jsonTypeInfo)
        => JsonSerializer.Deserialize(json, jsonTypeInfo)
            ?? throw new JsonException($"Failed to deserialize {jsonTypeInfo.Type.Name}.");

    public static ValueTask<T?> DeserializeAsync<T>(Stream utf8Json, JsonTypeInfo<T> jsonTypeInfo, CancellationToken cancellationToken = default)
        => JsonSerializer.DeserializeAsync(utf8Json, jsonTypeInfo, cancellationToken);

    private static JsonSerializerOptions CreateSerializerOptions()
    {
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault,
        };

        return options;
    }
}
