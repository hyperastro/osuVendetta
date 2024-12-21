using osuVendetta.Core.Anticheat.Data;

namespace osuVendetta.Core.Anticheat.Benchmark.Data;

public class AntiCheatBenchmarkDirectorySetting
{
    /// <summary>
    /// Directory containing the subdirectories and their replays
    /// </summary>
    public required DirectoryInfo Directory { get; init; }
    /// <summary>
    /// Processing the inference result to a usable benchmark result
    /// </summary>
    public required Func<AntiCheatModelResult, AntiCheatBenchmarkReplayResult> ResultProcessor { get; init; }
}
