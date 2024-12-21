namespace osuVendetta.Core.Anticheat.Benchmark.Data;

public class AntiCheatBenchmarkSettings
{
    /// <summary>
    /// Each directory setting is responsible for a batch of replays to benchmark
    /// </summary>
    public required AntiCheatBenchmarkDirectorySetting[] DirectorySettings { get; init; }
}
