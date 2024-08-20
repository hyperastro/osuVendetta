using Blazr.RenderState.WASM;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using MudBlazor.Services;
using osuvendetta.Web.Client.AntiCheat;
using osuvendetta.Core.AntiCheat;

namespace osuvendetta.Web.Client;

internal class Program
{
    static async Task Main(string[] args)
    {
        var builder = WebAssemblyHostBuilder.CreateDefault(args);

        builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

        builder.AddBlazrRenderStateWASMServices();

        builder.Services.AddTransient<IAntiCheatService, AntiCheatService>();
        builder.Services.AddTransient<IAntiCheatModelProvider, ModelProvider>();
        
        builder.Services.AddMudServices();

        await builder.Build().RunAsync();
    }
}
