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
            await MainMenu();
    }

    static async Task MainMenu()
    {
        const string PROCESS_FILE_OPTION = "Process File";
        const string PROCESS_FOLDER_OPTION = "Process Folder";
        const string EXIT_OPTION = "Exit";

        string result = AnsiConsole.Prompt(new SelectionPrompt<string>()
            .Title("osu!Vendetta Main Menu\n\nPick an option from the list below (arrow keys):")
            .AddChoices(new string[]
            {
                PROCESS_FILE_OPTION,
                PROCESS_FOLDER_OPTION,
                EXIT_OPTION
            }));

        switch (result)
        {
            case PROCESS_FILE_OPTION:
                await MenuProcessFile();
                break;

            case PROCESS_FOLDER_OPTION:
                await MenuProcessFolder();
                break;

            case EXIT_OPTION:
                MenuExit();
                break;
        }
    }

    static async Task MenuProcessFile()
    {
        const string FILE_PROMPT = "Enter the path to the file:";

        string filePath = AnsiConsole.Prompt(new TextPrompt<string>(FILE_PROMPT))
                          .Trim('"');

        while (!File.Exists(filePath))
        {
            Console.WriteLine($"Invalid path {filePath}");
            filePath = AnsiConsole.Prompt(new TextPrompt<string>(FILE_PROMPT))
                       .Trim('"');
        }

        await ProcessFiles(new List<FileInfo> { new FileInfo(filePath) });

        Console.WriteLine();
        Console.WriteLine("———");
        Console.WriteLine();
    }

    static async Task MenuProcessFolder()
    {
        const string DIRECTORY_PROMPT = "Enter the path to the folder containing the replays:";

        string folderPath = AnsiConsole.Prompt(new TextPrompt<string>(DIRECTORY_PROMPT))
                            .Trim('"');

        while (!Directory.Exists(folderPath))
        {
            Console.WriteLine($"Invalid path {folderPath}");
            folderPath = AnsiConsole.Prompt(new TextPrompt<string>(DIRECTORY_PROMPT))
                         .Trim('"');
        }

        await ProcessFiles(new DirectoryInfo(folderPath)
            .EnumerateFiles("*.osr", SearchOption.AllDirectories)
            .ToList());

        Console.WriteLine();
        Console.WriteLine("———");
        Console.WriteLine();
    }

    static void MenuExit()
    {
        Environment.Exit(0);
    }

    record class ModelOutput(string File, AntiCheatResult Result);

    static async Task ProcessFiles(List<FileInfo> files)
    {
        const string CSV_PATTERN = "\"{0}\",\"{1}\",\"{2}\",\"{3}\",\"{4}\",\"{5}\"";

        AntiCheatModelProvider provider = new AntiCheatModelProvider();
        AntiCheatService service = new AntiCheatService(provider);
        IAntiCheatModel model = await provider.LoadModelAsync(new AntiCheatProviderArgs());

        ConcurrentQueue<ModelOutput> modelOutput = new ConcurrentQueue<ModelOutput>();

        await Parallel.ForEachAsync(files, async (file, _) =>
        {
            try
            {

            List<ReplayFrame> replayFrames = ReplayDecoder.Decode(file.FullName).ReplayFrames;

            float[] inputData = service.ProcessReplayTokensNew(replayFrames);
            long[] dimensions = service.CreateModelDimensionsFor(inputData);

            AntiCheatResult result = await model.RunModelAsync(new InputArgs(inputData, dimensions));
            modelOutput.Enqueue(new ModelOutput(file.Name, result));

            Console.WriteLine($"Result for {file}:\n\t{result.Type} (Probability of cheating: {(100f - result.ProbabilityResult.ProbabilityNormal):N2}%) ({result.Message ?? string.Empty})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Something went wrong processing the file: {file.Name}:\n{ex}");
            }
        });

        StringBuilder output = new StringBuilder();
        output.AppendLine(string.Format(CSV_PATTERN, "File", "Result", "Player", "Probability Normal", "Probability Relax", "Message"));

        while (modelOutput.TryDequeue(out ModelOutput? result))
        {
            output.AppendLine(string.Format(CSV_PATTERN,
                result.File,
                result.Result.Type,
                result.Result.Metadata?.Player ?? "Unkown Player",
                result.Result.ProbabilityResult.ProbabilityNormal,
                result.Result.ProbabilityResult.ProbabilityRelax,
                result.Result.Message ?? string.Empty));
        }

        DateTime date = DateTime.Now;
        string csvName = $"{date:dd}.{date:MM}.{date:yyyy}-{date:HH}.{date:mm}-{date.Ticks}-AntiCheatReport.csv";

        File.WriteAllText(csvName, output.ToString());
        Console.WriteLine($"Saved report to: {Path.Combine(Environment.CurrentDirectory, csvName)}");
    }
}
