using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using osuVendetta.CLI.Menu.Pages;
using Spectre.Console;

namespace osuVendetta.CLI.Menu;

public class MenuSystem : IHostedService
{
    readonly Stack<MenuPage> _pageStack;
    readonly IServiceProvider _serviceProvider;
    readonly IHostApplicationLifetime _lifetime;

    public MenuSystem(IServiceProvider serviceProvider, IHostApplicationLifetime lifetime)
    {
        _pageStack = new Stack<MenuPage>();
        _serviceProvider = serviceProvider;
        _lifetime = lifetime;
    }

    /// <summary>
    /// Displays the menu
    /// </summary>
    /// <returns>True if we should exit</returns>
    /// <exception cref="InvalidOperationException"></exception>
    /// <exception cref="NullReferenceException"></exception>
    public async Task<bool> Display()
    {
        MenuPage currentPage = _pageStack.Peek();
        MenuPageResponse? pageResponse = await currentPage.Display();

        switch (pageResponse.ResponseType)
        {
            default:
                throw new InvalidOperationException($"Unkown response: {pageResponse.ResponseType}");

            case MenuPageResponseType.ToMainMenu:
                NavigateToMainMenu();
                break;

            case MenuPageResponseType.Retry:
                break;

            case MenuPageResponseType.PreviousPage:
                NavigateBack();
                break;

            case MenuPageResponseType.NextPage:
                if (pageResponse.NextMenuPageType is null)
                    throw new NullReferenceException("Next menu page cannot be null");

                NavigateTo(pageResponse.NextMenuPageType);
                break;

            case MenuPageResponseType.Exit:
                _lifetime.StopApplication();
                return true;
        }

        return false;
    }

    public void NavigateBack()
    {
        if (_pageStack.Count == 1)
            return;

        MenuPage page = _pageStack.Pop();
        page.Dispose();
    }

    public void NavigateTo(Type menuPageType)
    {
        IServiceScope serviceScope = _serviceProvider.CreateScope();
        MenuPage menuPage;

        try
        {
            object? menuPageObj = Activator.CreateInstance(menuPageType, serviceScope);

            if (menuPageObj is null)
                throw new InvalidOperationException($"Unable to create instance of type: {menuPageType.FullName}");

            if (menuPageObj is not MenuPage page)
                throw new InvalidOperationException($"Invalid constructor for menu page: {menuPageType.FullName}");

            menuPage = page;
        }
        catch (Exception)
        {
            serviceScope.Dispose();
            throw;
        }

        _pageStack.Push(menuPage);
    }

    public void NavigateTo<TMenuPage>()
        where TMenuPage : MenuPage
    {
        NavigateTo(typeof(TMenuPage));
    }

    public void NavigateToMainMenu()
    {
        while (_pageStack.Count > 1)
            _pageStack.Pop();
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        try
        {
            AnsiConsole.Foreground = Color.Cyan3;
            AnsiConsole.Background = Color.Black;

            NavigateTo<MainMenuPage>();

            bool shouldExit = false;
            while (!cancellationToken.IsCancellationRequested &&
                   !shouldExit)
            {
                shouldExit = await Display();
            }
        }
        finally
        {
            _lifetime.StopApplication();
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}
