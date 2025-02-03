using osuVendetta.AntiCheatModel128x3;
using osuVendetta.CLI.Components;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Configuration;
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

    public TrainingMenuPage(IAntiCheatModel model)
    {
        _model = model;
    }

    AntiCheatTrainer CreateTrainer()
    {
        CLIConfig config = BaseConfig.Load<CLIConfig>();

        DatasetArchive archiveTrain = DatasetArchive.Load(new FileInfo(config.TrainingDataset));
        ReplayDatasetProvider providerTrain = new ReplayDatasetProvider(archiveTrain, true);

        ReplayDatasetProvider providerTest;

        if (!string.IsNullOrEmpty(config.TestDataset))
        {
            DatasetArchive archiveTest = DatasetArchive.Load(new FileInfo(config.TestDataset));
            providerTest = new ReplayDatasetProvider(archiveTest, false);
        }
        else
        {
            providerTest = providerTrain;
        }

        TrainingProgressDisplay progressDisplay = new TrainingProgressDisplay();

        return new AntiCheatTrainer((AntiCheatModel128x3.AntiCheatModel128x3)_model,
            providerTrain, providerTrain,
            progressDisplay);
    }

    public override async Task<MenuPageResponse> DisplayAsync()
    {
        string title = string.Empty;

        double processedReplays = 0;
        double maxReplays = 0;
        DateTime start = DateTime.Now;
        double replaysPerSecond = 0;

        string originalConsoleTitle = Console.Title ?? string.Empty;

        Task rpsTask = new Task(async () =>
        {
            for (; ; )
            {
                if (maxReplays > 0 && processedReplays > 0)
                {
                    replaysPerSecond = processedReplays / (DateTime.Now - start).TotalSeconds;
                }

                Console.Title = $"{originalConsoleTitle} | Replays Per Second: {(int)replaysPerSecond}";
                await Task.Delay(500);
            }
        });


        AntiCheatTrainer trainer = CreateTrainer();
        Task trackerTask = trainer.TrainingTracker.DisplayAsync(CancellationToken.None);

        await Task.Run(() =>
        {
            CLIConfig config = BaseConfig.Load<CLIConfig>();

            rpsTask.Start();
            trainer.Train(9999, CancellationToken.None);
        });

        rpsTask.Dispose();
        trackerTask.Dispose();

        AnsiConsole.WriteLine("Press any key to return");
        Console.Title = originalConsoleTitle;
        Console.ReadLine();
        Console.ReadLine();

        return MenuPageResponse.ToMainMenu();
    }
}
