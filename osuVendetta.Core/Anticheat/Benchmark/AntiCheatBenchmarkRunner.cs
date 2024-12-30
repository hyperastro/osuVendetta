using OsuParsers.Decoders;
using OsuParsers.Replays;
using osuVendetta.Core.Anticheat.Benchmark.Data;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;

namespace osuVendetta.Core.Anticheat.Benchmark;

public interface IProgressReporter
{
    void Increment(double amount = 1.0);
    void SetMaxProgress(double max);
    void SetCurrentProgress(double current);
    void SetProgressTitle(string title);
}

public class ProgressReporter : IProgressReporter
{
    public Action<double> IncrementAction { get; init; }
    public Action<double> SetCurrentProgressAction { get; init; }
    public Action<double> SetMaxProgressAction { get; init; }
    public Action<string> SetProgressTitleAction { get; init; }

    public ProgressReporter(Action<double> incrementAction, 
                            Action<double> setCurrentProgressAction, 
                            Action<double> setMaxProgressAction, 
                            Action<string> setProgressTitleAction)
    {
        IncrementAction = incrementAction;
        SetCurrentProgressAction = setCurrentProgressAction;
        SetMaxProgressAction = setMaxProgressAction;
        SetProgressTitleAction = setProgressTitleAction;
    }

    public void Increment(double amount = 1)
    {
        IncrementAction(amount);
    }

    public void SetCurrentProgress(double current)
    {
        SetCurrentProgressAction(current);
    }

    public void SetMaxProgress(double max)
    {
        SetMaxProgressAction(max);
    }

    public void SetProgressTitle(string title)
    {
        SetProgressTitleAction(title);
    }
}

public class AntiCheatBenchmarkRunner : IAntiCheatBenchmarkRunner
{
    readonly IAntiCheatModel _antiCheatModel;
    readonly IReplayProcessor _replayProcessor;

    public AntiCheatBenchmarkRunner(IAntiCheatModel antiCheatModel, IReplayProcessor replayProcessor)
    {
        _antiCheatModel = antiCheatModel;
        _replayProcessor = replayProcessor;
    }

    public async Task<AntiCheatBenchmarkResult> Run(AntiCheatBenchmarkSettings settings, IProgressReporter progressReporter)
    {
        using FileStream modelStream = File.OpenRead("Data/128x3.safetensors");
        _antiCheatModel.Load(modelStream);

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
            await Parallel.ForEachAsync(files, async (replay, ctx) =>
            {
                AntiCheatBenchmarkReplayResult? result = await RunInference(replay, setting);
                progressReporter.Increment();

                if (result is null)
                    return;

                results.Add(result);
            });
        }

        return new AntiCheatBenchmarkResult(results.ToArray());
    }

    async Task<AntiCheatBenchmarkReplayResult?> RunInference(FileInfo file, AntiCheatBenchmarkDirectorySetting setting)
    {
        Replay replay = ReplayDecoder.Decode(file.FullName);
        ReplayValidationResult validation = _replayProcessor.IsValidReplay(replay);

        if (!validation.IsValid)
            return null;

        ReplayTokens tokens = await _replayProcessor.CreateTokensFromFramesAsync(file.Name, replay.ReplayFrames, true);
        AntiCheatModelResult result = _antiCheatModel.RunInference(tokens);

        return setting.ResultProcessor(result);
    }
}
