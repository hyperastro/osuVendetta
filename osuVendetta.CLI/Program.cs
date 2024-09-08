using OsuParsers.Decoders;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;
using osuVendetta.Core.AntiCheat.Data;
using System.Collections.Concurrent;
using System.Numerics;
using osuVendetta.CLI.AntiCheat;
using osuVendetta.Core.AntiCheat;
using Spectre.Console;

namespace osuVendetta.CLI;

internal partial class Program
{

    static SelectionPrompt<string> CreateMainMenuPrompt()
    {
        return new SelectionPrompt<string>()
            .Title("osu!Vendetta Main Menu\n\nPick an option from the list below (arrow keys):")
            .AddChoices(new string[]
            {
                "Process Files",
                "Process Folder",
                "Exit"
            });
    }

    static async Task Main(string[] args)
    {
        if (args.Length > 0)
        {
            List<FileInfo> filesToProcess = new List<FileInfo>();

            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i].ToLower();

                switch (arg)
                {
                    case "-file":
                        FileInfo file = new FileInfo(args[++i]);

                        if (!file.Exists)
                        {
                            Console.WriteLine($"File {file.FullName} not found");
                            break;
                        }

                        filesToProcess.Add(file);
                        break;

                    case "-folder":
                        DirectoryInfo dir = new DirectoryInfo(args[++i]);

                        if (!dir.Exists)
                        {
                            Console.WriteLine($"Directory {dir.FullName} not found");
                            break;
                        }

                        filesToProcess.AddRange(dir.EnumerateFiles("*.osr", SearchOption.AllDirectories));
                        break;
                }
            }

            if (filesToProcess.Count == 0)
            {
                Console.WriteLine("No files to process found");
                return;
            }

            Console.WriteLine($"Processing {filesToProcess.Count} files...");
            await ProcessFiles(filesToProcess);

            Console.WriteLine();
            Console.WriteLine("———");
            Console.WriteLine();
        }

        for (; ; )
        {
            string result = AnsiConsole.Prompt(CreateMainMenuPrompt());
            
            switch (result)
            {
                case "Process Files":
                    string filePath = AnsiConsole.Prompt(
                        new TextPrompt<string>("Enter the path to the file:"))
                        .Trim('"');

                    if (!File.Exists(filePath))
                    {
                        Console.WriteLine("Invalid path, press any key to continue...");
                        Console.ReadKey();
                        Console.Clear();
                        break;
                    }

                    await ProcessFiles(new List<FileInfo> { new FileInfo(filePath) });
                    Console.WriteLine();
                    Console.WriteLine("———");
                    Console.WriteLine();
                    break;

                case "Process Folder":
                    string folderPath = AnsiConsole.Prompt(
                        new TextPrompt<string>("Enter the path to the file:"))
                        .Trim('"');

                    if (!Directory.Exists(folderPath))
                    {
                        Console.WriteLine("Invalid path, press any key to continue...");
                        Console.ReadKey();
                        Console.Clear();
                        break;
                    }

                    await ProcessFiles(new DirectoryInfo(folderPath)
                        .EnumerateFiles("*.osr", SearchOption.AllDirectories)
                        .ToList());
                    Console.WriteLine();
                    Console.WriteLine("———");
                    Console.WriteLine();
                    break;

                case "Exit":
                    Environment.Exit(0);
                    break;
            }
        }

    }

    static async Task ProcessFiles(List<FileInfo> files)
    {
        AntiCheatModelProvider provider = new AntiCheatModelProvider();
        AntiCheatService service = new AntiCheatService(provider);
        IAntiCheatModel model = await provider.LoadModelAsync(new AntiCheatProviderArgs());

        foreach (FileInfo file in files)
        {
            List<ReplayFrame> replayFrames = ReplayDecoder.Decode(file.FullName).ReplayFrames;

            float[] inputData = service.ProcessReplayTokensNew(replayFrames);
            long[] dimensions = service.CreateModelDimensionsFor(inputData);

            AntiCheatResult result = await model.RunModelAsync(new InputArgs(inputData, dimensions));

            Console.WriteLine($"Result for {file}:\n\t{result.Type} ({result.Message})");
        }
    }
}
