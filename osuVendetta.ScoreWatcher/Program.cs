using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using osuVendetta.AntiCheatModel128x3;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.OsuApi;
using osuVendetta.ScoreWatcher.Services;
using Spectre.Console;

namespace osuVendetta.ScoreWatcher;

internal class Program
{
    public static HttpClient HttpClient { get; } = new HttpClient();

    public static IHost? App { get; private set; }

    static async Task Main(string[] args)
    {
        AnsiConsole.Foreground = ConsoleColor.Cyan;
        AnsiConsole.WriteLine("Starting...");

        HostApplicationBuilder builder = Host.CreateApplicationBuilder();

        builder.Logging.ClearProviders();
        builder.Logging.SetMinimumLevel(LogLevel.Information);
        //builder.Logging.AddConsole(options =>
        //{
        //    options.LogToStandardErrorThreshold = LogLevel.Error;
        //});
        builder.Logging.AddSimpleConsole(options =>
        {
            options.SingleLine = true;
            options.TimestampFormat = "HH:mm:ss:fff ";
            options.IncludeScopes = false;
        });

        ConfigureServices(builder.Services);

        App = builder.Build();

        AppDomain.CurrentDomain.UnhandledException += (sender, eventArgs) =>
        {
            Exception exception = (Exception)eventArgs.ExceptionObject;

            ILogger logger = App.Services.GetRequiredService<ILogger<Program>>();
            logger.LogCritical($"An unhandled exception occured:\n{exception.Message}");
        };

        await App.RunAsync();
    }

    static void ConfigureServices(IServiceCollection services)
    {
        services.AddLogging();
        services.AddSingleton<HttpClient>();
        services.AddSingleton<IOsuApiClient, OsuApiClient>();
        services.AddSingleton<IDiscordWebhookService, DiscordWebhookService>();

        services.AddSingleton<IReplayProcessor, ReplayProcessor>();
        services.AddSingleton<IAntiCheatModel, AntiCheatModel128x3.AntiCheatModel128x3>();

        services.AddHostedService<ScoreWatcherService>();
    }
}
