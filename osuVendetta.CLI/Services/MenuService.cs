using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using osuVendetta.CLI.Menu;
using osuVendetta.CLI.Menu.Pages;
using Spectre.Console;

namespace osuVendetta.CLI.Services;

/// <summary>
/// Service responsible for handling the menu
/// </summary>
public class MenuService : IHostedService
{
    readonly Stack<MenuPage> _pageStack;
    readonly IServiceProvider _serviceProvider;
    readonly IHostApplicationLifetime _lifetime;

    public MenuService(IServiceProvider serviceProvider, IHostApplicationLifetime lifetime)
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

        AnsiConsole.Clear();

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

    /// <summary>
    /// Navigates to the previous page
    /// </summary>
    public void NavigateBack()
    {
        if (_pageStack.Count == 1)
            return;

        MenuPage page = _pageStack.Pop();
        page.Dispose();
    }

    /// <summary>
    /// <inheritdoc cref="NavigateTo{TMenuPage}()"/>
    /// </summary>
    /// <param name="menuPageType">Menu page type</param>
    /// <exception cref="InvalidOperationException"></exception>
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

    /// <summary>
    /// Navigates to the next page
    /// </summary>
    /// <typeparam name="TMenuPage">Menu page type</typeparam>
    public void NavigateTo<TMenuPage>()
        where TMenuPage : MenuPage
    {
        NavigateTo(typeof(TMenuPage));
    }

    /// <summary>
    /// Returns to the main menu
    /// </summary>
    public void NavigateToMainMenu()
    {
        while (_pageStack.Count > 1)
            _pageStack.Pop();
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _lifetime.ApplicationStarted.Register(async () => await RunMenuAsync());
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }

    /// <summary>
    /// Runs the menu loop
    /// </summary>
    /// <param name="cancellationToken"></param>
    /// <returns></returns>
    async Task RunMenuAsync()
    {
        // let the system fire up fully before running the system, not sure if there is a better way
        await Task.Delay(15);

        Console.WriteLine("RunMenu");

        try
        {
            AnsiConsole.Foreground = Color.Cyan3;
            AnsiConsole.Background = Color.Black;

            NavigateTo<MainMenuPage>();

            bool shouldExit = false;
            while (!shouldExit)
            {
                shouldExit = await Display();
            }
        }
        finally
        {
            _lifetime.StopApplication();
        }
    }
}
