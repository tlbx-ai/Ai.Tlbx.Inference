using Microsoft.Extensions.DependencyInjection;

namespace Ai.Tlbx.Inference.Configuration;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddAiInference(this IServiceCollection services, Action<AiInferenceOptions> configure)
    {
        var options = new AiInferenceOptions();
        configure(options);
        services.AddSingleton(options);
        services.AddHttpClient<IAiInferenceClient, AiInferenceClient>();
        return services;
    }
}
