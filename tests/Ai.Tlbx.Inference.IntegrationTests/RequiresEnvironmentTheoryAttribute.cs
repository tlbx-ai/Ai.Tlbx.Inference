namespace Ai.Tlbx.Inference.IntegrationTests;

[AttributeUsage(AttributeTargets.Method)]
internal sealed class RequiresEnvironmentTheoryAttribute : TheoryAttribute
{
    private const int DefaultTimeoutMilliseconds = 180_000;

    public RequiresEnvironmentTheoryAttribute(string variableName)
    {
        VariableName = variableName;
        Timeout = DefaultTimeoutMilliseconds;

        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(variableName)))
        {
            Skip = $"Integration test requires environment variable {variableName}.";
        }
    }

    public string VariableName { get; }
}
