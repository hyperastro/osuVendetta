using Microsoft.Extensions.DependencyInjection;
using Spectre.Console.Cli;

namespace osuVendetta.CLI;

public class TypeRegistrar : ITypeRegistrar
{
    public IServiceCollection Services { get; }

    public TypeRegistrar(IServiceCollection services)
    {
        Services = services;
    }

    public TypeRegistrar() : this(new ServiceCollection())
    {

    }

    public ITypeResolver Build()
    {
        return new TypeResolver(Services.BuildServiceProvider());
    }

    public void Register(Type service, Type implementation)
    {
        Services.AddScoped(service, implementation);
    }

    public void RegisterInstance(Type service, object implementation)
    {
        Services.AddSingleton(service, implementation);
    }

    public void RegisterLazy(Type service, Func<object> factory)
    {
        Services.AddSingleton(service, factory);
    }
}
