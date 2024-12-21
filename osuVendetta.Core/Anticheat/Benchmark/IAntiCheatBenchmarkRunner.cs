using osuVendetta.Core.Anticheat.Benchmark.Data;
using osuVendetta.Core.AntiCheat;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.Anticheat.Benchmark;

public interface IAntiCheatBenchmarkRunner
{
    /// <summary>
    /// Run the benchmark
    /// </summary>
    /// <param name="settings">Benchmark settings</param>
    /// <returns>Benchmark result</returns>
    Task<AntiCheatBenchmarkResult> Run(AntiCheatBenchmarkSettings settings, IProgressReporter progressReporter);
}
