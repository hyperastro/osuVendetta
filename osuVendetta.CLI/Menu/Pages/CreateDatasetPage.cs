using osuVendetta.Core.IO.Dataset;
using osuVendetta.Core.Replays;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.CLI.Menu.Pages;

public class CreateDatasetPage : MenuPage
{
    readonly IReplayProcessor _replayProcessor;

    public CreateDatasetPage(IReplayProcessor replayProcessor)
    {
        _replayProcessor = replayProcessor;
    }

    public override async Task<MenuPageResponse> DisplayAsync()
    {
        string normalDatasetPath = new TextPrompt<string>("Folder path:")
            .PromptFor(
                s => !string.IsNullOrEmpty(s),
                "Normal dataset folder");

        string relaxDatasetPath = new TextPrompt<string>("Folder path:")
            .PromptFor(
                s => !string.IsNullOrEmpty(s),
                "Relax dataset folder");

        string destinationPath = new TextPrompt<string>("Archive path:")
            .PromptFor(
                s => !string.IsNullOrEmpty(s),
                "Archive output path");

        Dictionary<ReplayDatasetClass, List<FileInfo>> datasets = new Dictionary<ReplayDatasetClass, List<FileInfo>>
        {
            {
                ReplayDatasetClass.Normal,
                GatherFiles(normalDatasetPath)
            },
            {
                ReplayDatasetClass.Relax,
                GatherFiles(relaxDatasetPath)
            }
        };

        AnsiConsole.Clear();
        AnsiConsole.WriteLine("Creating dataset...");

        Progress progress = AnsiConsole.Progress();

        await progress.StartAsync(async context =>
        {
            int totalEntries = datasets.Values.Sum(d => d.Count);

            ProgressTask progressTask = context.AddTask("Replay", maxValue: totalEntries);
            ProgressTask progressScalersTask = context.AddTask("Scalers", maxValue: totalEntries * 2);

            DatasetArchive.Create(new FileInfo(destinationPath),
                datasets, _replayProcessor, progressTask, progressScalersTask);
        });


        AnsiConsole.WriteLine($"Dataset created");
        AnsiConsole.WriteLine("Press 'Enter' to return");
        Console.ReadLine();

        return MenuPageResponse.ToMainMenu();
    }

    List<FileInfo> GatherFiles(string path)
    {
        DirectoryInfo dir = new DirectoryInfo(path);

        List<FileInfo> files = new List<FileInfo>();
        files.AddRange(dir.EnumerateFiles("*.osr"));
        files.AddRange(dir.EnumerateFiles("*.txt"));

        return files;
    } 
}
