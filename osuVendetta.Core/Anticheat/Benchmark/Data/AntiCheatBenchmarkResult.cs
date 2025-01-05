namespace osuVendetta.Core.Anticheat.Benchmark.Data;

public class AntiCheatBenchmarkDirectoryResult
{
    /// <summary>
    /// Benchmark results of all replays
    /// </summary>
    public AntiCheatBenchmarkReplayResult[] ReplayResults { get; init; }
    /// <summary>
    /// Location of the replays
    /// </summary>
    public DirectoryInfo Directory { get; init; }
    /// <summary>
    /// Average accuracy of all replays
    /// </summary>
    public float Accuracy { get; init; }

    public AntiCheatBenchmarkDirectoryResult(DirectoryInfo directory, AntiCheatBenchmarkReplayResult[] replayResults)
    {
        ReplayResults = replayResults;
        Directory = directory;

        if (replayResults.Length > 0)
        {
            for (int i = 0; i < replayResults.Length; i++)
                Accuracy += ReplayResults[i].Accuracy;

            Accuracy /= replayResults.Length;
        }
    }

    public override string ToString()
    {
        return $"Total Results: {ReplayResults.Length}, Accuracy: {Accuracy}";
    }
}

public class AntiCheatBenchmarkResult
{
    /// <summary>
    /// Benchmark results of all replays
    /// </summary>
    public AntiCheatBenchmarkDirectoryResult[] DirectoryResults { get; init; }
    /// <summary>
    /// Average accuracy of all replays
    /// </summary>
    public float Accuracy { get; init; }

    public AntiCheatBenchmarkResult(AntiCheatBenchmarkDirectoryResult[] directoryResults)
    {
        DirectoryResults = directoryResults;

        if (directoryResults.Length > 0)
        {
            for (int i = 0; i < directoryResults.Length; i++)
                Accuracy += DirectoryResults[i].Accuracy;

            Accuracy /= directoryResults.Length;
        }
    }

    public override string ToString()
    {
        return $"Total Results: {DirectoryResults.Length}, Accuracy: {Accuracy}";
    }
}
