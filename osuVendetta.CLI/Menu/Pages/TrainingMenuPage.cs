using osuVendetta.AntiCheatModel128x3;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO.Dataset;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Utility;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

namespace osuVendetta.CLI.Menu.Pages;

public class TrainingMenuPage : MenuPage
{
    readonly IAntiCheatModel _model;
    readonly IReplayProcessor _replayProcessor;

    public TrainingMenuPage(IAntiCheatModel model, IReplayProcessor replayProcessor)
    {
        _model = model;
        _replayProcessor = replayProcessor;
    }

    public override async Task<MenuPageResponse> DisplayAsync()
    {
        double current = 0;
        double max = 0;
        string title = string.Empty;

        AntiCheatTrainer trainer = new AntiCheatTrainer((AntiCheatModel128x3.AntiCheatModel128x3)_model);

        double processedReplays = 0;
        double maxReplays = 0;
        DateTime start = DateTime.Now;
        double replaysPerSecond = 0;

        Task rpsTask = new Task(async () =>
        {
            for (; ; )
            {
                if (maxReplays > 0 && processedReplays > 0)
                {
                    replaysPerSecond = processedReplays / (DateTime.Now - start).TotalSeconds;
                }

                Console.Title = $"Replays Per Second: {replaysPerSecond}";
                await Task.Delay(500);
            }
        });

        await AnsiConsole.Progress().StartAsync(async progressContext =>
        {
            ProgressTask progress = progressContext.AddTask("Training...");
            ProgressReporter progressReporter = new ProgressReporter(
                value =>
                {
                    progress.Increment(value);
                    processedReplays += value;

                    if (processedReplays >= 40)
                    {
                        processedReplays = 0;
                        start = DateTime.Now;
                    }
                },
                value => progress.Value(processedReplays = value),
                value => progress.MaxValue(maxReplays = value),
                title => progress.Description(title));

            rpsTask.Start();

            await Task.Run(() =>
            {
                trainer.RunTraining(new ReplayDatasetProvider(
                    _replayProcessor,
                    [
                        new ReplayDatasetInfo(ReplayDatasetClass.Normal, new DirectoryInfo("C:\\Users\\Unk\\Downloads\\DatasetV1\\DatasetV1\\normal").EnumerateFiles("*.txt", SearchOption.AllDirectories).ToList()),
                    new ReplayDatasetInfo(ReplayDatasetClass.Relax, new DirectoryInfo("C:\\Users\\Unk\\Downloads\\DatasetV1\\DatasetV1\\relax").EnumerateFiles("*.txt", SearchOption.AllDirectories).ToList())
                    ]),
                    progressReporter);
            });

            rpsTask.Dispose();
        });

        AnsiConsole.WriteLine("Press any key to return");
        Console.ReadLine();
        Console.ReadLine();

        return MenuPageResponse.ToMainMenu();
    }
}
