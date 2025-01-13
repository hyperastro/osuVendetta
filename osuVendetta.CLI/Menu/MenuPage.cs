using Microsoft.Extensions.DependencyInjection;

namespace osuVendetta.CLI.Menu;

public abstract class MenuPage
{
    public string? Title { get; set; }

    public abstract Task<MenuPageResponse> Display();
}
