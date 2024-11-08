using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using MudBlazor.Services;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Web.Client.Services;

namespace osuVendetta.Web.Client;

internal class Program
{
    static async Task Main(string[] args)
    {
        var builder = WebAssemblyHostBuilder.CreateDefault(args);

        builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
        builder.Services.AddMudServices();

        builder.Services.AddScoped<IAntiCheatModel, AntiCheatModel>();
        builder.Services.AddScoped<IAntiCheatConfigProvider, AntiCheatConfigProvider>();
        builder.Services.AddScoped<IAntiCheatService, AntiCheatService>();

        await builder.Build().RunAsync();
    }
}
