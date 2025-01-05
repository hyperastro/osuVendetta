using Microsoft.Extensions.DependencyInjection;

namespace osuVendetta.CLI.Menu;

public abstract class MenuPage : IDisposable
{
    public string? Title { get; set; }
    public bool IsDisposed { get; private set; }

    protected readonly IServiceScope _serviceScope;

    public MenuPage(IServiceScope serviceScope)
    {
        _serviceScope = serviceScope;
    }

    public abstract Task<MenuPageResponse> Display();

    public void Dispose()
    {
        if (IsDisposed)
            return;

        _serviceScope.Dispose();
        IsDisposed = true;
    }
}
