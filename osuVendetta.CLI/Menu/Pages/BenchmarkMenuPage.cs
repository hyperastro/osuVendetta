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

    public BenchmarkMenuPage(IAntiCheatBenchmarkRunner antiCheatBenchmarkRunner)
    {
        _antiCheatBenchmarkRunner = antiCheatBenchmarkRunner;
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
            .AddColumns(new TableColumn("Path"),
                        new TableColumn("Processed Replays")
                            .Alignment(Justify.Right),
                        new TableColumn("Accuracy")
                            .Alignment(Justify.Right));

        int totalFiles = 0;

        if (result is not null)
        {
            for (int i = 0; i < result.DirectoryResults.Length; i++)
            {
                string accuracyStr = (result.DirectoryResults[i].Accuracy * 100).ToString("N2").PadLeft(6, ' ');

                table.AddRow($"{result.DirectoryResults[i].Directory.FullName}",
                             $"{result.DirectoryResults[i].ReplayResults.Length}",
                             $"{accuracyStr} %");

                totalFiles += result.DirectoryResults[i].ReplayResults.Length;
            }
        }

        AnsiConsole.Write(table);

        table = new Table()
            .AddColumns(new TableColumn("Time total (in seconds)")
                            .Alignment(Justify.Right),
                        new TableColumn("Time average (in seconds")
                            .Alignment(Justify.Right),
                        new TableColumn("Total Processed Replays")
                            .Alignment(Justify.Right))
            .AddRow($"{stopwatch.Elapsed.TotalSeconds:n4}",
                    $"{(stopwatch.Elapsed.TotalSeconds / totalFiles):n4}",
                    $"{totalFiles}");

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
                },
                new AntiCheatBenchmarkDirectorySetting
                {
                    Directory = new DirectoryInfo("C:/osu!/Replays"),
                    ResultProcessor = m => ProcessBenchmark(m, false)
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