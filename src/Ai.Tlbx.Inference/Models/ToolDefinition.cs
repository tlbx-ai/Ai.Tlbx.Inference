using System.Text.Json;

namespace Ai.Tlbx.Inference;

public sealed record ToolDefinition
{
    public required string Name { get; init; }
    public required string Description { get; init; }
    public required JsonElement ParametersSchema { get; init; }
}
