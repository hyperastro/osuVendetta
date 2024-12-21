using Microsoft.Extensions.DependencyInjection;
using Spectre.Console.Cli;

namespace osuVendetta.CLI;

public class TypeResolver : ITypeResolver
{
    public IServiceProvider ServiceProvider { get; }

    public TypeResolver(IServiceProvider serviceProvider)
    {
        ServiceProvider = serviceProvider;
    }

    public object? Resolve(Type? type)
    {
        ArgumentNullException.ThrowIfNull(type);
        return ServiceProvider.GetRequiredService(type);
    }
}