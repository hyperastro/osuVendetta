using MudBlazor.Services;
using OsuParsers.Replays;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using osuVendetta.CoreLib.AntiCheat.Data;
using osuVendetta.Web.Client.Pages;
using osuVendetta.Web.Client.Services;
using osuVendetta.Web.Components;

namespace osuVendetta.Web;

public class AntiCheatModelServer : IAntiCheatModel
{
    public Task LoadAsync(AntiCheatModelLoadArgs loadArgs)
    {
        throw new NotImplementedException();
    }

    public Task<Logit> Run(ModelInput input)
    {
        throw new NotImplementedException();
    }

    public Task UnloadAsync()
    {
        throw new NotImplementedException();
    }
}
public class AntiCheatServiceServer : IAntiCheatService
{
    public Task<BaseAntiCheatResult> ProcessReplayAsync(Replay replay, bool runInParallel)
    {
        throw new NotImplementedException();
    }
}
public class AntiCheatConfigProviderServer : IAntiCheatConfigProvider
{
    public Task<AntiCheatConfig?> GetConfig()
    {
        throw new NotImplementedException();
    }

    public Task<string?> GetModelPath()
    {
        throw new NotImplementedException();
    }
}



public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Add MudBlazor services
        builder.Services.AddMudServices();

        // Add services to the container.
        builder.Services.AddRazorComponents()
            .AddInteractiveServerComponents()
            .AddInteractiveWebAssemblyComponents();

        builder.Services.AddScoped<IAntiCheatModel, AntiCheatModelServer>();
        builder.Services.AddScoped<IAntiCheatConfigProvider, AntiCheatConfigProviderServer>();
        builder.Services.AddScoped<IAntiCheatService, AntiCheatServiceServer>();

        builder.Services.AddControllers();

        builder.Services.AddHttpClient();

        var app = builder.Build();

        // Configure the HTTP request pipeline.
        if (app.Environment.IsDevelopment())
        {
            app.UseWebAssemblyDebugging();
        }
        else
        {
            app.UseExceptionHandler("/Error");
            // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
            app.UseHsts();
        }

        app.UseCors("AllowAll");
        app.UseHttpsRedirection();

        app.UseStaticFiles();
        app.UseAntiforgery();

        app.MapRazorComponents<App>()
            .AddInteractiveServerRenderMode()
            .AddInteractiveWebAssemblyRenderMode()
            .AddAdditionalAssemblies(typeof(Client._Imports).Assembly);

        app.MapControllers();

        app.Run();
    }
}
