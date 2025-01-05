using OsuParsers.Decoders;
using OsuParsers.Replays;
using osuVendetta.Core.Anticheat.Benchmark.Data;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using osuVendetta.Core.Utility;

namespace osuVendetta.Core.Anticheat.Benchmark;

public class AntiCheatBenchmarkRunner : IAntiCheatBenchmarkRunner
{
    readonly IAntiCheatModel _antiCheatModel;
    readonly IReplayProcessor _replayProcessor;

    public AntiCheatBenchmarkRunner(IAntiCheatModel antiCheatModel, IReplayProcessor replayProcessor)
    {
        _antiCheatModel = antiCheatModel;
        _replayProcessor = replayProcessor;
    }

    public AntiCheatBenchmarkResult Run(AntiCheatBenchmarkSettings settings, IProgressReporter progressReporter)
    {
        List<AntiCheatBenchmarkReplayResult> results = new List<AntiCheatBenchmarkReplayResult>();
        Dictionary<DirectoryInfo, (AntiCheatBenchmarkDirectorySetting, List<FileInfo>)> replays = new Dictionary<DirectoryInfo, (AntiCheatBenchmarkDirectorySetting, List<FileInfo>)>();

        int totalFiles = 0;

        for (int i = 0; i < settings.DirectorySettings.Length; i++)
        {
            List<FileInfo> files = new List<FileInfo>();

            foreach (FileInfo file in settings.DirectorySettings[i].Directory.EnumerateFiles("*.osr", SearchOption.AllDirectories))
                files.Add(file);

            replays[settings.DirectorySettings[i].Directory] = (settings.DirectorySettings[i], files);
            totalFiles += files.Count;
        }

        progressReporter.SetMaxProgress(totalFiles);
        progressReporter.SetProgressTitle($"Processing replays (total: {totalFiles})");

        foreach ((AntiCheatBenchmarkDirectorySetting setting, List<FileInfo> files) in replays.Values)
        {
            Parallel.ForEach(files, (replay, ctx) =>
            {
                AntiCheatBenchmarkReplayResult? result = RunInference(replay, setting);
                progressReporter.Increment();

                if (result is null)
                    return;

                results.Add(result);
            });
        }

        return new AntiCheatBenchmarkResult(results.ToArray());
    }

    AntiCheatBenchmarkReplayResult? RunInference(FileInfo file, AntiCheatBenchmarkDirectorySetting setting)
    {
        // TODO: replaydecoder closes the file after reading once? need to check
        FileStream replayStream = File.OpenRead(file.FullName);
        try
        {
            ReplayValidationResult validation = _replayProcessor.IsValidReplay(replayStream);

            if (!validation.IsValid)
                return null;
        }
        finally
        {
            replayStream.Dispose();
        }
        
        try
        {
            replayStream = File.OpenRead(file.FullName);
            ReplayTokens tokens = _replayProcessor.CreateTokensParallel(replayStream);
            AntiCheatModelResult result = _antiCheatModel.RunInference(tokens);

            return setting.ResultProcessor(result);
        }
        finally
        {
            replayStream.Dispose();
        }

    }
}
