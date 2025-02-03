using Spectre.Console;

namespace osuVendetta.CLI.Menu.Pages;

public static class ConsoleUtility
{
    public static TResult PromptFor<TResult>(this IPrompt<TResult> prompt, Func<TResult, bool> condition, string? title = null,  bool clearLine = true)
    {
        TResult result;

        do
        {
            if (clearLine)
                AnsiConsole.Clear();

            if (title is not null)
                AnsiConsole.WriteLine(title);

            result = AnsiConsole.Prompt(prompt);
        }
        while (!condition(result));

        return result;
    }
}
