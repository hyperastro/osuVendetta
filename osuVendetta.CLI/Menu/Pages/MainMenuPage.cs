using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using osuVendetta.Core.AntiCheat;
using Spectre.Console;

namespace osuVendetta.CLI.Menu.Pages;

public class MainMenuPage : MenuPage
{
    const string _TITLE = "Welcome to osu!Vendetta";
    const string _RUN_INFERENCE_CHOICE = "Run Inference";
    const string _RUN_BENCHMARK_CHOICE = "Run Benchmark";
    const string _SETTINGS_CHOICE = "Settings";
    const string _EXIT_CHOICE = "Exit";

    readonly IAntiCheatModel _antiCheatModel;

    public MainMenuPage(IServiceScope serviceScope) : base(serviceScope)
    {
        _antiCheatModel = serviceScope.ServiceProvider.GetService<IAntiCheatModel>()
            ?? throw new InvalidOperationException("AntiCheatModel not registered as service");
    }

    public override async Task<MenuPageResponse> Display()
    {
        RunDebug();

        string result = AnsiConsole.Prompt(new SelectionPrompt<string>()
            .Title(_TITLE)
            .AddChoices(_RUN_INFERENCE_CHOICE,
                        _RUN_BENCHMARK_CHOICE,
                        _SETTINGS_CHOICE,
                        _EXIT_CHOICE));

        switch (result)
        { 
            case _RUN_INFERENCE_CHOICE:
                return MenuPageResponse.NextPage<InferenceMenuPage>();

            case _RUN_BENCHMARK_CHOICE:
                return MenuPageResponse.NextPage<BenchmarkMenuPage>();

            case _SETTINGS_CHOICE:
                throw new NotImplementedException();

            case _EXIT_CHOICE:
                return MenuPageResponse.Exit();

            default:
                return MenuPageResponse.Retry();
        }
    }

    void RunDebug()
    {
        //using FileStream modelStream = File.OpenRead("Data/128x3.bin");
        //_antiCheatModel.Load(modelStream);

        float[,,] tokens = new float[1, _antiCheatModel.Config.StepsPerChunk, _antiCheatModel.Config.FeaturesPerStep];

        int steps = 0;
        foreach (string line in File.ReadAllLines("Data/modelInput.csv"))
        {
            string[] lineSplit = line.Split(',');

            for (int i = 0; i < _antiCheatModel.Config.FeaturesPerStep; i++)
            {
                if (!float.TryParse(lineSplit[i], out float token))
                {
                    Console.WriteLine($"Unable to parse {lineSplit[i]}");
                    continue;
                }

                tokens[0, steps, i] = token;
            }

            steps++;
        }

        _antiCheatModel.RunInference(new Core.Replays.Data.ReplayTokens
        {
            ReplayName = "Debug Replay",
            Tokens = tokens
        });
    }
}
