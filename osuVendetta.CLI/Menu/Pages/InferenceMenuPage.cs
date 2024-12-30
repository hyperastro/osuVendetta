using Microsoft.Extensions.DependencyInjection;
using OsuParsers.Decoders;
using OsuParsers.Replays;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.CLI.Menu.Pages;

public class InferenceMenuPage : MenuPage
{
    readonly IAntiCheatModel _antiCheatModel;
    readonly IReplayProcessor _replayProcessor;

    public InferenceMenuPage(IServiceScope serviceScope) : base(serviceScope)
    {
        _antiCheatModel = serviceScope.ServiceProvider.GetService<IAntiCheatModel>()
            ?? throw new InvalidOperationException("AntiCheatModel not registered as service");

        _replayProcessor = serviceScope.ServiceProvider.GetService<IReplayProcessor>()
            ?? throw new InvalidOperationException("Replay processor not registered as service");
    }

    public override async Task<MenuPageResponse> Display()
    {
        List<FileInfo> replaysToProcess = PromptForReplays();
        List<AntiCheatModelResult> processedReplays = await ProcessReplays(replaysToProcess);

        AnsiConsole.WriteLine("Preparing result...");

        Table table = new Table()
            .Title("Replay Results")
            .AddColumns("Filename", "Segment Accuracy");

        for (int i = 0; i < processedReplays.Count; i++)
        {
            AntiCheatModelResult result = processedReplays[i];

            table.AddRow(result.ReplayName.Replace("[", "[[")
                                          .Replace("]", "]]"));

            float total = 0;
            for (int j = 0; j < result.Segments.Length; j++)
            {
                total += result.Segments[j];

                table.AddRow(string.Empty,
                             $"{(result.Segments[j] * 100):N2} %");
            }

            table.AddRow(string.Empty,
                         $"Average: {(total / result.Segments.Length * 100):N2} %");
        }

        AnsiConsole.Clear();

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine("Press \"Enter\" to return");

        Console.ReadLine();

        return MenuPageResponse.PreviousPage();
    }

    async Task<List<AntiCheatModelResult>> ProcessReplays(List<FileInfo> replays)
    {
        using FileStream modelStream = File.OpenRead("Data/128x3.safetensors");
        _antiCheatModel.Load(modelStream);

        List<AntiCheatModelResult> results = new List<AntiCheatModelResult>();

        await Parallel.ForEachAsync(replays, async (replayFile, ctx) =>
        {
            Replay replay = ReplayDecoder.Decode(replayFile.FullName);
            ReplayValidationResult validation = _replayProcessor.IsValidReplay(replay);

            if (!validation.IsValid)
                return;

            ReplayTokens tokens = await _replayProcessor.CreateTokensFromFramesAsync(replayFile.Name, replay.ReplayFrames, true);
            AntiCheatModelResult result = _antiCheatModel.RunInference(tokens);

            results.Add(result);
        });

        return results;
    }

    List<FileInfo> PromptForReplays()
    {
        List<FileInfo> files = new List<FileInfo>();

        while (files.Count == 0)
        {
            string path = PromptPath();

            if (string.IsNullOrEmpty(path))
                continue;

            path = path.Trim('"');

            if (Directory.Exists(path))
            {
                DirectoryInfo dir = new DirectoryInfo(path);
                files.AddRange(dir.EnumerateFiles("*.osr"));
            }
            else if (File.Exists(path))
            {
                FileInfo file = new FileInfo(path);

                if (file.Extension.Equals(".osr", StringComparison.CurrentCultureIgnoreCase))
                    files.Add(new FileInfo(path));
            }
        }

        return files;
    }

    string PromptPath()
    {
        AnsiConsole.Clear();
        AnsiConsole.WriteLine("Select the folder or file you want to run inference on");

        return AnsiConsole.Prompt(new TextPrompt<string>("File path:"));
    }
}
