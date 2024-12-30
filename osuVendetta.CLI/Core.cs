
using Google.Protobuf.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using osuVendetta.CLI.Menu;
using osuVendetta.Core.Anticheat.Benchmark;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO;
using osuVendetta.Core.Replays;
using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics;

namespace osuVendetta.CLI;

internal static class Program
{
    public static IHost? App { get; private set; }
    public static ICommandApp? CommandApp { get; private set; }

    static async Task Main(string[] args)
    {
        if (args.Length > 0)
            await RunCommandApp(args);
        else
            await RunAppHost();
    }

    static async Task RunCommandApp(string[] args)
    {
        IServiceCollection services = new ServiceCollection();
        ConfigureServices(services, false);

        TypeRegistrar registrar = new TypeRegistrar(services);

        CommandApp = new CommandApp(registrar);
        CommandApp.Configure(config =>
        {

        });

        int exitCode = await CommandApp.RunAsync(args);

        if (exitCode != 0)
        {
            Environment.Exit(exitCode);
            return;
        }
    }

    static async Task RunAppHost()
    {
        HostApplicationBuilder builder = Host.CreateApplicationBuilder();
        ConfigureServices(builder.Services, true);

        App = builder.Build();

        await App.RunAsync();
    }

    static void ConfigureServices(IServiceCollection services, bool includeMenuSystem)
    {
        if (includeMenuSystem)
            services.AddHostedService<MenuSystem>();

        services.AddSingleton<IAntiCheatModel, AntiCheatModel128x3.AntiCheatModel128x3>();
        services.AddSingleton<IReplayProcessor, ReplayProcessor>();
        services.AddSingleton<IAntiCheatBenchmarkRunner, AntiCheatBenchmarkRunner>();
    }
}
