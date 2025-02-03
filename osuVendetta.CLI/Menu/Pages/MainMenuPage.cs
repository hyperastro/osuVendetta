using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using osuVendetta.Core.AntiCheat;
using Spectre.Console;

namespace osuVendetta.CLI.Menu.Pages;

public class MainMenuPage : MenuPage
{
    const string _TITLE = "Welcome to osu!Vendetta";
    const string _RUN_INFERENCE_CHOICE = "Inference";
    const string _RUN_BENCHMARK_CHOICE = "Benchmark";
    const string _TRAININGS_CHOICE = "Training";
    const string _SETTINGS_CHOICE = "Settings";
    const string _CREATE_DATASET_CHOICE = "Create Dataset";
    const string _EXIT_CHOICE = "Exit";

    string? _errorMessage;

    public MainMenuPage()
    {
    }

    public override async Task<MenuPageResponse> DisplayAsync()
    {
        if (!string.IsNullOrEmpty(_errorMessage))
        {
            AnsiConsole.WriteLine(_errorMessage);
            _errorMessage = null;
        }

        string result = AnsiConsole.Prompt(new SelectionPrompt<string>()
            .Title(_TITLE)
            .AddChoices(_RUN_INFERENCE_CHOICE,
                        _RUN_BENCHMARK_CHOICE,
                        _TRAININGS_CHOICE,
                        _CREATE_DATASET_CHOICE,
                        _SETTINGS_CHOICE,
                        _EXIT_CHOICE));


        switch (result)
        { 
            case _RUN_INFERENCE_CHOICE:
                return MenuPageResponse.NextPage<InferenceMenuPage>();

            case _RUN_BENCHMARK_CHOICE:
                return MenuPageResponse.NextPage<BenchmarkMenuPage>();

            case _SETTINGS_CHOICE:
                _errorMessage = "Settings not yet implemented";
                return MenuPageResponse.Retry();

            case _TRAININGS_CHOICE:
                return MenuPageResponse.NextPage<TrainingMenuPage>();

            case _CREATE_DATASET_CHOICE:
                return MenuPageResponse.NextPage<CreateDatasetPage>();

            case _EXIT_CHOICE:
                return MenuPageResponse.Exit();

            default:
                return MenuPageResponse.Retry();
        }
    }
}
