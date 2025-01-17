using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using Perfolizer.Horology;

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

        Summary[] summaries = BenchmarkRunner.Run(typeof(Program).Assembly, config);

        Console.WriteLine("Press 'Enter' to exit");
        Console.ReadLine();
    }
}
