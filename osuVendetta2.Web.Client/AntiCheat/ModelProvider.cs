using Microsoft.JSInterop;
using osuVendetta.Core.AntiCheat;

namespace osuVendetta.Web.Client.AntiCheat;

public record class ModelProviderArgs(
    string PathToModel,
    string JsInteropFunction
) : AntiCheatModelProviderArgs;

public class ModelProvider : IAntiCheatModelProvider
{
    readonly IJSRuntime _jsRuntime;
    readonly string _jsLoadFunc = "antiCheat.load";

    public ModelProvider(IJSRuntime jsRuntime)
    {
        _jsRuntime = jsRuntime;
    }

    public async Task<IAntiCheatModel> LoadModelAsync(AntiCheatModelProviderArgs args)
    {
        ModelProviderArgs modelArgs = (ModelProviderArgs)args;
        await _jsRuntime.InvokeVoidAsync(_jsLoadFunc, modelArgs.PathToModel);

        return new AntiCheatModel(_jsRuntime);
    }
}
