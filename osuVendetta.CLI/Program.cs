using OsuParsers.Decoders;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;
//using osuVendetta.Core.AntiCheat.Data;
using System.Collections.Concurrent;
using System.Numerics;
using osuVendetta.CLI.AntiCheat;
//using osuVendetta.Core.AntiCheat;
using Spectre.Console;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.CoreLib.AntiCheat.Data;
using osuVendetta.Core.Replays.Data;
using osuVendetta.Core.AntiCheat.Data;

namespace osuVendetta.CLI;

internal partial class Program
{
    static AntiCheatConfig? _antiCheatConfig;
    static AntiCheatRunner? _antiCheatRunner;

    static bool _isAntiCheatLoadedOnRun;
    static bool _runInParallel;

    static async Task Main(string[] args)
    {
        _isAntiCheatLoadedOnRun = await LoadAntiCheatAsync();

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

                    case "-parallel":
                        _runInParallel = true;
                        break;
                }
            }

            if (filesToProcess.Count == 0)
            {
                Console.WriteLine("No files to process found");
                return;
            }

            Console.WriteLine($"Processing {filesToProcess.Count} files...");
            await ProcessFiles(filesToProcess, _runInParallel);

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

        string extraInfo = string.Empty;

        if (!_isAntiCheatLoadedOnRun)
            extraInfo = "(AntiCheat config not found, you will be asked to create or specify one in a later step)\n";

        string result = AnsiConsole.Prompt(new SelectionPrompt<string>()
            .Title($"osu!Vendetta Main Menu\n\n{extraInfo}Pick an option from the list below (arrow keys):")
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

        await MenuProcess(new List<FileInfo> { new FileInfo(filePath) });
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

        await MenuProcess(new DirectoryInfo(folderPath)
            .EnumerateFiles("*.osr", SearchOption.AllDirectories)
            .ToList());
    }

    static async Task MenuProcess(List<FileInfo> files)
    {
        const string RUN_IN_PARALLEL_PROMPT = "Do you want to run in parallel? (multithreaded)";
        const string NEW_CONFIG_PATH_PROMPT = "No config found, please specify a custom path or leave it empty to create one";
        const string CREATE_CONFIG_PROMPT = "No config found, do you want to create a default one?";
        const string CONFIG_SAVE_PATH = "anticheat.config";

        bool runInParallel = AnsiConsole.Prompt(new TextPrompt<bool>(RUN_IN_PARALLEL_PROMPT));

        if (!_isAntiCheatLoadedOnRun)
        {
            string configPath = AnsiConsole.Prompt(new TextPrompt<string>(NEW_CONFIG_PATH_PROMPT)
            {
                AllowEmpty = true
            });

            if (string.IsNullOrEmpty(configPath) || !File.Exists(configPath) ||
                !await LoadAntiCheatAsync(configPath))
            {
                bool createDefaultConfig = AnsiConsole.Prompt(new TextPrompt<bool>(CREATE_CONFIG_PROMPT));

                if (createDefaultConfig)
                {
                    AntiCheatConfig defaultConfig = AntiCheatModel192x2.CreateDefaultConfigForModel();
                    File.WriteAllText(CONFIG_SAVE_PATH, defaultConfig.ToJson());

                    Console.WriteLine($"Config created and saved to {CONFIG_SAVE_PATH}");
                    Console.ReadKey();
                    Environment.Exit(1);
                }

                Console.WriteLine("Config not found, press any key to exit...");
                Console.ReadKey();
                Environment.Exit(1);
            }
        }

        Console.WriteLine("Running anticheat");

        await ProcessFiles(files, runInParallel);

        Console.WriteLine();
        Console.WriteLine("———");
        Console.WriteLine();
    }


    static void MenuExit()
    {
        Environment.Exit(0);
    }

    record class ModelOutput(string File, Replay Replay, AntiCheatResult Result, string ResultType);

    static async Task ProcessFiles(List<FileInfo> files, bool runInParallel)
    {
        const string CSV_PATTERN = "\"{0}\",\"{1}\",\"{2}\",\"{3}\",\"{4}\"";

        // TODO: ask to load or create default config if config not found
        ArgumentNullException.ThrowIfNull(_antiCheatRunner);

        //IAntiCheatModel model = new AntiCheatModel192x2();
        //ReplayProcessor replayProcessor = new ReplayProcessor(_antiCheatConfig);
        ConcurrentQueue<ModelOutput> modelOutput = new ConcurrentQueue<ModelOutput>();

        await Parallel.ForEachAsync(files, async (file, _) =>
        {
            Replay replay = ReplayDecoder.Decode(file.FullName);
            BaseAntiCheatResult antiCheatResult = await _antiCheatRunner.ProcessReplayAsync(replay, runInParallel);

            if (antiCheatResult is AntiCheatResult result)
            {
                string resultType = "Normal";
                double probabilityCheating = 100f - result.CheatProbability.Normal;

                if (result.CheatProbability.Relax >= 0.5)
                    resultType = "Relax";

                modelOutput.Enqueue(new ModelOutput(file.Name, replay, result, resultType));

                Console.WriteLine($"Result for {file}:\n\t{resultType} (Probability of cheating: {probabilityCheating:N2}%)");
            }
            else
            {
                Console.WriteLine($"Something went wrong processing the file: {file.Name}:\n{antiCheatResult}");
            }
        });

        StringBuilder output = new StringBuilder();
        output.AppendLine(string.Format(CSV_PATTERN, "File", "Result", "Player", "Probability Normal", "Probability Relax", "Message"));

        while (modelOutput.TryDequeue(out ModelOutput? result))
        {
            output.AppendLine(string.Format(CSV_PATTERN,
                result.File,
                result.ResultType,
                result.Replay.PlayerName ?? "Unkown Player",
                result.Result.CheatProbability.Normal,
                result.Result.CheatProbability.Relax));
        }

        DateTime date = DateTime.Now;
        string csvName = $"{date:dd}.{date:MM}.{date:yyyy}-{date:HH}.{date:mm}-{date.Ticks}-AntiCheatReport.csv";

        File.WriteAllText(csvName, output.ToString());
        Console.WriteLine($"Saved report to: {Path.Combine(Environment.CurrentDirectory, csvName)}");
    }

    static async Task<bool> LoadAntiCheatAsync(string configPath = "anticheat.config", string modelPath = "192x2.onnx")
    {
        if (!File.Exists(configPath) || !File.Exists(modelPath))
            return false;

        AntiCheatConfig? config = AntiCheatConfig.FromJson(File.ReadAllText(configPath));

        if (config is null)
            return false;

        AntiCheatModel192x2 model = new AntiCheatModel192x2();
        await model.LoadAsync(new AntiCheatModelLoadArgs
        {
            AntiCheatConfig = config,
            ModelBytes = File.ReadAllBytes(modelPath)
        });

        _antiCheatConfig = config;
        _antiCheatRunner = new AntiCheatRunner(model, config);

        return true;
    }
}
