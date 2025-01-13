
using Google.Protobuf.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using osuVendetta.CLI.Services;
using osuVendetta.Core.Anticheat.Benchmark;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO;
using osuVendetta.Core.Replays;
using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;
using System.Diagnostics;
using Tensorboard;
using TorchSharp;
using static TorchSharp.torch;

namespace osuVendetta.CLI;

internal class Program
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
        AnsiConsole.Foreground = ConsoleColor.Cyan;
        AnsiConsole.WriteLine("Starting...");

        HostApplicationBuilder builder = Host.CreateApplicationBuilder();
        ConfigureServices(builder.Services, true);

        builder.Logging.ClearProviders();
        builder.Logging.SetMinimumLevel(LogLevel.Information);
        builder.Logging.AddConsole(options =>
        {
            options.LogToStandardErrorThreshold = LogLevel.Error;
        });
        
        App = builder.Build();

        AppDomain.CurrentDomain.UnhandledException += (sender, eventArgs) =>
        {
            Exception exception = (Exception)eventArgs.ExceptionObject;

            ILogger logger = App.Services.GetRequiredService<ILogger<Program>>();
            logger.LogCritical($"An unhandled exception occured:\n{exception.Message}");
        };

        await App.RunAsync();
    }

    static void ConfigureServices(IServiceCollection services, bool includeMenuSystem)
    {
        services.AddSingleton<IAntiCheatModel, AntiCheatModel128x3.AntiCheatModel128x3>();
        services.AddSingleton<IReplayProcessor, ReplayProcessor>();
        services.AddSingleton<IAntiCheatBenchmarkRunner, AntiCheatBenchmarkRunner>();

        services.AddHostedService<AntiCheatService>();

        if (includeMenuSystem)
            services.AddHostedService<MenuService>();
    }
}
