namespace Ai.Tlbx.Inference.Tests.Models;

public sealed class TokenUsageTests
{
    [Fact]
    public void TotalTokens_SumsInputOutputAndThinking()
    {
        var usage = new TokenUsage
        {
            InputTokens = 100,
            OutputTokens = 50,
            ThinkingTokens = 25,
        };

        Assert.Equal(175, usage.TotalTokens);
    }

    [Fact]
    public void TotalTokens_ExcludesCacheTokens()
    {
        var usage = new TokenUsage
        {
            InputTokens = 100,
            OutputTokens = 50,
            CacheReadTokens = 30,
            CacheWriteTokens = 20,
        };

        Assert.Equal(150, usage.TotalTokens);
    }

    [Fact]
    public void OperatorPlus_AccumulatesAllFields()
    {
        var a = new TokenUsage
        {
            InputTokens = 100,
            OutputTokens = 50,
            CacheReadTokens = 10,
            CacheWriteTokens = 5,
            ThinkingTokens = 20,
        };

        var b = new TokenUsage
        {
            InputTokens = 200,
            OutputTokens = 100,
            CacheReadTokens = 15,
            CacheWriteTokens = 8,
            ThinkingTokens = 30,
        };

        var result = a + b;

        Assert.Equal(300, result.InputTokens);
        Assert.Equal(150, result.OutputTokens);
        Assert.Equal(25, result.CacheReadTokens);
        Assert.Equal(13, result.CacheWriteTokens);
        Assert.Equal(50, result.ThinkingTokens);
    }

    [Fact]
    public void Default_AllZero()
    {
        var usage = new TokenUsage();

        Assert.Equal(0, usage.InputTokens);
        Assert.Equal(0, usage.OutputTokens);
        Assert.Equal(0, usage.CacheReadTokens);
        Assert.Equal(0, usage.CacheWriteTokens);
        Assert.Equal(0, usage.ThinkingTokens);
        Assert.Equal(0, usage.TotalTokens);
    }
}
