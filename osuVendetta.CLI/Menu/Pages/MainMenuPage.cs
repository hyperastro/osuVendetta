using Microsoft.Extensions.DependencyInjection;
using Spectre.Console;

namespace osuVendetta.CLI.Menu.Pages;

public class MainMenuPage : MenuPage
{
    const string _TITLE = "Welcome to osu!Vendetta";
    const string _RUN_INFERENCE_CHOICE = "Run Inference";
    const string _RUN_BENCHMARK_CHOICE = "Run Benchmark";
    const string _SETTINGS_CHOICE = "Settings";
    const string _EXIT_CHOICE = "Exit";

    public MainMenuPage(IServiceScope serviceScope) : base(serviceScope)
    {

    }

    public override async Task<MenuPageResponse> Display()
    {
        AnsiConsole.Clear();

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
        }

        return MenuPageResponse.Retry();
    }
}
