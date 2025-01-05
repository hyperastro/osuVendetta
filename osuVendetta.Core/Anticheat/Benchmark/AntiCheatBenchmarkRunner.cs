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
        //List<AntiCheatBenchmarkReplayResult> results = new List<AntiCheatBenchmarkReplayResult>();
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

        //Dictionary<DirectoryInfo, AntiCheatBenchmarkDirectoryResult> results = new Dictionary<DirectoryInfo, AntiCheatBenchmarkDirectoryResult>();
        List<AntiCheatBenchmarkDirectoryResult> results = new List<AntiCheatBenchmarkDirectoryResult>();

        foreach (DirectoryInfo dir in replays.Keys)
        {
            (AntiCheatBenchmarkDirectorySetting setting, List<FileInfo> files) = replays[dir];
            List<AntiCheatBenchmarkReplayResult> replayResults = new List<AntiCheatBenchmarkReplayResult>();

            Parallel.ForEach(files, replayFile =>
            {
                AntiCheatBenchmarkReplayResult? result = RunInference(replayFile, setting);
                progressReporter.Increment();

                if (result is null)
                    return;

                replayResults.Add(result);
            });

            results.Add(new AntiCheatBenchmarkDirectoryResult(dir, replayResults.ToArray()));
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
