using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace Ai.Tlbx.Inference.Schema;

internal static class JsonSchemaGenerator
{
    private static readonly ConcurrentDictionary<Type, string> _cache = new();

    [RequiresUnreferencedCode("JSON schema generation uses reflection. For AOT, provide CompletionRequest.JsonSchema directly.")]
    [RequiresDynamicCode("JSON schema generation uses reflection. For AOT, provide CompletionRequest.JsonSchema directly.")]
    public static JsonElement Generate<T>() => Generate(typeof(T));

    [RequiresUnreferencedCode("JSON schema generation uses reflection. For AOT, provide CompletionRequest.JsonSchema directly.")]
    [RequiresDynamicCode("JSON schema generation uses reflection. For AOT, provide CompletionRequest.JsonSchema directly.")]
    public static JsonElement Generate(Type type)
    {
        var json = GenerateAsString(type);
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.Clone();
    }

    [RequiresUnreferencedCode("JSON schema generation uses reflection. For AOT, provide CompletionRequest.JsonSchema directly.")]
    [RequiresDynamicCode("JSON schema generation uses reflection. For AOT, provide CompletionRequest.JsonSchema directly.")]
    public static string GenerateAsString(Type type)
    {
        return _cache.GetOrAdd(type, static t =>
        {
            using var stream = new MemoryStream();
            using var writer = new Utf8JsonWriter(stream);
            WriteSchema(writer, t);
            writer.Flush();
            return Encoding.UTF8.GetString(stream.ToArray());
        });
    }

    [UnconditionalSuppressMessage("Trimming", "IL2070", Justification = "Callers are annotated with RequiresUnreferencedCode.")]
    private static void WriteSchema(Utf8JsonWriter writer, Type type)
    {
        type = Nullable.GetUnderlyingType(type) ?? type;

        if (type == typeof(string))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "string");
            writer.WriteEndObject();
            return;
        }

        if (type == typeof(bool))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "boolean");
            writer.WriteEndObject();
            return;
        }

        if (type == typeof(int) || type == typeof(long))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "integer");
            writer.WriteEndObject();
            return;
        }

        if (type == typeof(float) || type == typeof(double) || type == typeof(decimal))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "number");
            writer.WriteEndObject();
            return;
        }

        if (type == typeof(DateTime) || type == typeof(DateTimeOffset))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "string");
            writer.WriteString("format", "date-time");
            writer.WriteEndObject();
            return;
        }

        if (type == typeof(Guid))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "string");
            writer.WriteString("format", "uuid");
            writer.WriteEndObject();
            return;
        }

        if (type.IsEnum)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "string");
            writer.WritePropertyName("enum");
            writer.WriteStartArray();
            foreach (var name in Enum.GetNames(type))
            {
                writer.WriteStringValue(name);
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
            return;
        }

        var elementType = GetEnumerableElementType(type);
        if (elementType is not null)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "array");
            writer.WritePropertyName("items");
            WriteSchema(writer, elementType);
            writer.WriteEndObject();
            return;
        }

        writer.WriteStartObject();
        writer.WriteString("type", "object");

        var properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
        var requiredProps = new List<string>();

        writer.WritePropertyName("properties");
        writer.WriteStartObject();

        foreach (var prop in properties)
        {
            if (!prop.CanRead)
                continue;

            var propName = JsonNamingPolicy.CamelCase.ConvertName(prop.Name);
            writer.WritePropertyName(propName);

            var propType = prop.PropertyType;
            var isNullable = Nullable.GetUnderlyingType(propType) is not null;
            var nullabilityContext = new NullabilityInfoContext();
            var nullabilityInfo = nullabilityContext.Create(prop);
            var isReferenceNullable = !propType.IsValueType && nullabilityInfo.WriteState == NullabilityState.Nullable;

            if (!isNullable && !isReferenceNullable)
            {
                requiredProps.Add(propName);
            }

            WriteSchema(writer, propType);
        }

        writer.WriteEndObject();

        if (requiredProps.Count > 0)
        {
            writer.WritePropertyName("required");
            writer.WriteStartArray();
            foreach (var req in requiredProps)
            {
                writer.WriteStringValue(req);
            }
            writer.WriteEndArray();
        }

        writer.WriteEndObject();
    }

    [UnconditionalSuppressMessage("Trimming", "IL2070", Justification = "Callers are annotated with RequiresUnreferencedCode.")]
    private static Type? GetEnumerableElementType(Type type)
    {
        if (type.IsArray)
            return type.GetElementType();

        if (type.IsGenericType)
        {
            var genericDef = type.GetGenericTypeDefinition();
            if (genericDef == typeof(List<>) ||
                genericDef == typeof(IList<>) ||
                genericDef == typeof(IReadOnlyList<>) ||
                genericDef == typeof(IEnumerable<>) ||
                genericDef == typeof(ICollection<>) ||
                genericDef == typeof(IReadOnlyCollection<>))
            {
                return type.GetGenericArguments()[0];
            }
        }

        foreach (var iface in type.GetInterfaces())
        {
            if (iface.IsGenericType && iface.GetGenericTypeDefinition() == typeof(IEnumerable<>))
            {
                if (type != typeof(string))
                    return iface.GetGenericArguments()[0];
            }
        }

        return null;
    }
}
