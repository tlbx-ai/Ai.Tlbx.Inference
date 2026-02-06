using System.Text.Json;
using Ai.Tlbx.Inference.Schema;

namespace Ai.Tlbx.Inference.Tests.Schema;

public sealed class JsonSchemaGeneratorTests
{
    [Fact]
    public void Generate_String_ReturnsStringType()
    {
        var schema = JsonSchemaGenerator.Generate<string>();
        Assert.Equal("string", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Int_ReturnsIntegerType()
    {
        var schema = JsonSchemaGenerator.Generate<int>();
        Assert.Equal("integer", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Long_ReturnsIntegerType()
    {
        var schema = JsonSchemaGenerator.Generate<long>();
        Assert.Equal("integer", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Double_ReturnsNumberType()
    {
        var schema = JsonSchemaGenerator.Generate<double>();
        Assert.Equal("number", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Float_ReturnsNumberType()
    {
        var schema = JsonSchemaGenerator.Generate<float>();
        Assert.Equal("number", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Decimal_ReturnsNumberType()
    {
        var schema = JsonSchemaGenerator.Generate<decimal>();
        Assert.Equal("number", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Bool_ReturnsBooleanType()
    {
        var schema = JsonSchemaGenerator.Generate<bool>();
        Assert.Equal("boolean", schema.GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_DateTime_ReturnsStringWithDateTimeFormat()
    {
        var schema = JsonSchemaGenerator.Generate<DateTime>();
        Assert.Equal("string", schema.GetProperty("type").GetString());
        Assert.Equal("date-time", schema.GetProperty("format").GetString());
    }

    [Fact]
    public void Generate_Guid_ReturnsStringWithUuidFormat()
    {
        var schema = JsonSchemaGenerator.Generate<Guid>();
        Assert.Equal("string", schema.GetProperty("type").GetString());
        Assert.Equal("uuid", schema.GetProperty("format").GetString());
    }

    [Fact]
    public void Generate_Enum_ReturnsStringWithEnumValues()
    {
        var schema = JsonSchemaGenerator.Generate<TestEnum>();
        Assert.Equal("string", schema.GetProperty("type").GetString());

        var enumValues = schema.GetProperty("enum")
            .EnumerateArray()
            .Select(e => e.GetString())
            .ToList();

        Assert.Contains("Alpha", enumValues);
        Assert.Contains("Beta", enumValues);
        Assert.Contains("Gamma", enumValues);
    }

    [Fact]
    public void Generate_ListOfString_ReturnsArrayWithStringItems()
    {
        var schema = JsonSchemaGenerator.Generate<List<string>>();
        Assert.Equal("array", schema.GetProperty("type").GetString());
        Assert.Equal("string", schema.GetProperty("items").GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_ArrayOfInt_ReturnsArrayWithIntegerItems()
    {
        var schema = JsonSchemaGenerator.Generate<int[]>();
        Assert.Equal("array", schema.GetProperty("type").GetString());
        Assert.Equal("integer", schema.GetProperty("items").GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Poco_ReturnsObjectWithProperties()
    {
        var schema = JsonSchemaGenerator.Generate<TestPoco>();
        Assert.Equal("object", schema.GetProperty("type").GetString());

        var props = schema.GetProperty("properties");
        Assert.Equal("string", props.GetProperty("name").GetProperty("type").GetString());
        Assert.Equal("integer", props.GetProperty("age").GetProperty("type").GetString());
    }

    [Fact]
    public void Generate_Poco_RequiredContainsNonNullableProperties()
    {
        var schema = JsonSchemaGenerator.Generate<TestPoco>();
        var required = schema.GetProperty("required")
            .EnumerateArray()
            .Select(e => e.GetString())
            .ToList();

        Assert.Contains("name", required);
        Assert.Contains("age", required);
    }

    [Fact]
    public void Generate_PocoWithNullable_NullablePropertyNotRequired()
    {
        var schema = JsonSchemaGenerator.Generate<TestPocoWithNullable>();
        var required = schema.GetProperty("required")
            .EnumerateArray()
            .Select(e => e.GetString())
            .ToList();

        Assert.Contains("name", required);
        Assert.DoesNotContain("description", required);
    }

    [Fact]
    public void Generate_NullableInt_ReturnsIntegerType()
    {
        var schema = JsonSchemaGenerator.Generate<int?>();
        Assert.Equal("integer", schema.GetProperty("type").GetString());
    }

    public enum TestEnum
    {
        Alpha,
        Beta,
        Gamma
    }

    public sealed class TestPoco
    {
        public string Name { get; set; } = "";
        public int Age { get; set; }
    }

    public sealed class TestPocoWithNullable
    {
        public string Name { get; set; } = "";
        public string? Description { get; set; }
    }
}
