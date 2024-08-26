using Blazr.RenderState.Server;
using MudBlazor.Services;
using osuVendetta.Web.Components;

namespace osuVendetta.Web;

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

        builder.Services.AddControllers();

        builder.Services.AddHttpClient();

        builder.AddBlazrRenderStateServerServices();
        //builder.Services.AddCors(options =>
        //{
        //    options.AddPolicy("AllowAll", builder =>
        //    {
        //        builder.AllowAnyOrigin()
        //                .AllowAnyMethod()
        //                .AllowAnyHeader();
        //    });
        //});

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
