using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Mathematics;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using osuVendetta.CLI;
using osuVendetta.Core.Configuration;
using Perfolizer.Horology;
using Spectre.Console;

namespace osuVendetta.Metrics;

internal class Program
{
    static void Main(string[] args)
    {
        IConfig config = DefaultConfig.Instance
            .AddJob(Job.Default
#if RELEASE_WIN_CUDA
                .WithCustomBuildConfiguration("Release-Win-Cuda")
#elif RELEASE_WIN_CPU
                .WithCustomBuildConfiguration("Release-Win-Cpu")
#elif RELEASE_LINUX_CUDA
                .WithCustomBuildConfiguration("Release-Linux-Cuda")
#elif RELEASE_LINUX_CPU                                        
                .WithCustomBuildConfiguration("Release-Linux-Cpu")
#endif
                .AsDefault());

        Table table = new Table()
            .Title("Benchmark results")
            .AddColumns("Name", "Mean", "Error", "Std Dev");

        Summary[] summaries = BenchmarkRunner.Run(typeof(Program).Assembly, config);

        foreach (Summary summary in summaries)
        {
            foreach (BenchmarkReport report in summary.Reports)
            {
                Statistics stats = report.ResultStatistics!;

                table.AddRow(report.BenchmarkCase.Descriptor.WorkloadMethod.Name,
                             $"{(stats.Mean / 1_000_000):n3} ms",
                             $"{(stats.StandardError / 1_000_000):n3} ms",
                             $"{(stats.StandardDeviation  / 1_000_000):n3} ms");
            }
        }

        AnsiConsole.Foreground = ConsoleColor.Cyan;

        AnsiConsole.WriteLine();
        AnsiConsole.WriteLine("---------------------------");
        AnsiConsole.WriteLine();

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        AnsiConsole.WriteLine("Press 'Enter' to exit");
        Console.ReadLine();
    }
}
