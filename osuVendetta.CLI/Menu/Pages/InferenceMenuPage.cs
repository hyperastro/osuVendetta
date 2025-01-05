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
    class InferenceReplayResult
    {
        public required string ReplayName { get; set; }
        public required AntiCheatModelResult InferenceResult { get; set; }
    }

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
        List<InferenceReplayResult> processedReplays = ProcessReplays(replaysToProcess);

        AnsiConsole.WriteLine("Preparing result...");

        Table table = new Table()
            .Title("Replay Results")
            .AddColumns("Filename", "Segment Accuracy");

        for (int i = 0; i < processedReplays.Count; i++)
        {
            InferenceReplayResult result = processedReplays[i];

            table.AddRow(result.ReplayName.Replace("[", "[[")
                                          .Replace("]", "]]"));

            float total = 0;
            for (int j = 0; j < result.InferenceResult.Segments.Length; j++)
            {
                total += result.InferenceResult.Segments[j];

                table.AddRow(string.Empty,
                             $"{(result.InferenceResult.Segments[j] * 100):N2} %");
            }

            table.AddRow(string.Empty,
                         $"Average: {(total / result.InferenceResult.Segments.Length * 100):N2} %");
        }

        AnsiConsole.Clear();

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine("Press \"Enter\" to return");

        Console.ReadLine();

        return MenuPageResponse.PreviousPage();
    }

    List<InferenceReplayResult> ProcessReplays(List<FileInfo> replays)
    {
        List<InferenceReplayResult> results = new List<InferenceReplayResult>();

        Parallel.ForEach(replays, (replayFile, ctx) =>
        {
            // TODO: replaydecoder closes the file after reading once? need to check
            FileStream replayStream = File.OpenRead(replayFile.FullName);
            try
            {
                ReplayValidationResult validation = _replayProcessor.IsValidReplay(replayStream);

                if (!validation.IsValid)
                    return;
            }
            finally
            {
                replayStream.Dispose();
            }

            try
            {
                replayStream = File.OpenRead(replayFile.FullName);
                ReplayTokens tokens = _replayProcessor.CreateTokensParallel(replayStream);
                AntiCheatModelResult result = _antiCheatModel.RunInference(tokens);

                results.Add(new InferenceReplayResult
                {
                    InferenceResult = result,
                    ReplayName = replayFile.Name
                });
            }
            finally
            {
                replayStream.Dispose();
            }
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
