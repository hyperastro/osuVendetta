using Microsoft.Extensions.DependencyInjection;
using osuVendetta.Core.Anticheat.Benchmark;
using osuVendetta.Core.Anticheat.Benchmark.Data;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.Utility;
using Spectre.Console;
using System.Diagnostics;
using System.Diagnostics.Tracing;

namespace osuVendetta.CLI.Menu.Pages;

public class BenchmarkMenuPage : MenuPage
{
    readonly IAntiCheatBenchmarkRunner _antiCheatBenchmarkRunner;

    public BenchmarkMenuPage(IServiceScope serviceScope) : base(serviceScope)
    {
        _antiCheatBenchmarkRunner = serviceScope.ServiceProvider.GetService<IAntiCheatBenchmarkRunner>()
            ?? throw new InvalidOperationException("Benchmark runner not registered as service");
    }

    public override async Task<MenuPageResponse> Display()
    {
        AnsiConsole.WriteLine("Running benchmark...");

        AntiCheatBenchmarkResult? result = null;

        Stopwatch stopwatch = new Stopwatch();
        await AnsiConsole.Progress().StartAsync(async progressContext =>
        {
            ProgressTask progress = progressContext.AddTask("Processing Replays");
            ProgressReporter progressReporter = new ProgressReporter(
                value => progress.Increment(value),
                value => progress.Value(value),
                value => progress.MaxValue(value),
                title => progress.Description(title));

            AntiCheatBenchmarkSettings settings = CreateDefaultSettings();

            stopwatch.Start();
            result = await Task.Run(() => _antiCheatBenchmarkRunner.Run(settings, progressReporter));
            stopwatch.Stop();
        });

        AnsiConsole.Clear();
        AnsiConsole.WriteLine("Finished benchmark");

        Table table = new Table()
            .Title("Benchmark Result")
            .AddColumns(new TableColumn("Total Processed Replays"),
                        new TableColumn("Accuracy"),
                        new TableColumn("Time Total (seconds)"),
                        new TableColumn("Time Average (seconds)"));

        if (result is not null)
            table.AddRow($"{result.ReplayResults.Length}", 
                         $"{(result.Accuracy * 100):n2} %",
                         $"{stopwatch.Elapsed.TotalSeconds:n4}",
                         $"{(stopwatch.Elapsed.TotalSeconds / result.ReplayResults.Length):n4}");

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine("Press \"Enter\" to return");

        Console.ReadLine();

        return MenuPageResponse.PreviousPage();
    }

    AntiCheatBenchmarkSettings CreateDefaultSettings()
    {
        return new AntiCheatBenchmarkSettings
        {
            DirectorySettings = [
                new AntiCheatBenchmarkDirectorySetting
                {
                    Directory = new DirectoryInfo("Data/Benchmark/Normal"),
                    ResultProcessor = m => ProcessBenchmark(m, false)
                },
                new AntiCheatBenchmarkDirectorySetting
                {
                    Directory = new DirectoryInfo("Data/Benchmark/Relax"),
                    ResultProcessor = m => ProcessBenchmark(m, true)
                }
            ]
        };
    }

    AntiCheatBenchmarkReplayResult ProcessBenchmark(AntiCheatModelResult modelResult, bool isRelax)
    {
        // count all correct segments to get average of correct/incorrect 
        float accuracy = modelResult.Segments.Where(segment => isRelax ? segment >= .5f : segment < .5f)
                                             .Count();

        accuracy /= modelResult.Segments.Length;

        return new AntiCheatBenchmarkReplayResult
        {
            Accuracy = accuracy
        };
    }
}