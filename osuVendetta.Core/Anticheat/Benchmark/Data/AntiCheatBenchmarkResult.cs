namespace osuVendetta.Core.Anticheat.Benchmark.Data;

public class AntiCheatBenchmarkResult
{
    /// <summary>
    /// Benchmark results of all replaysa
    /// </summary>
    public AntiCheatBenchmarkReplayResult[] ReplayResults { get; init; }
    /// <summary>
    /// Average accuracy of all replays
    /// </summary>
    public float Accuracy { get; init; }

    public AntiCheatBenchmarkResult(AntiCheatBenchmarkReplayResult[] replayResults)
    {
        ReplayResults = replayResults;

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
