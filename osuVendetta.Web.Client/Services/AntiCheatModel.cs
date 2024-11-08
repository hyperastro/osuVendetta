using Microsoft.JSInterop;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using System.Text;

namespace osuVendetta.Web.Client.Services;

public class AntiCheatModel : IAntiCheatModel
{
    readonly static string _onnxInteropJsFunc = "antiCheat.run";
    readonly static string _onnxLoadJsFunc = "antiCheat.load";
    readonly static string _onnxUnloadJsFunc = "antiCheat.unload";

    readonly IJSRuntime _jsRuntime;

    bool _isLoaded;

    public AntiCheatModel(IJSRuntime jsRuntime)
    {
        _jsRuntime = jsRuntime;
    }

    public async Task LoadAsync(AntiCheatModelLoadArgs loadArgs)
    {
        if (_isLoaded) 
            throw new InvalidOperationException("Model already loaded");

        await _jsRuntime.InvokeVoidAsync(_onnxLoadJsFunc, Encoding.UTF8.GetString(loadArgs.ModelBytes));
        _isLoaded = true;
    }

    public async Task<Logit> Run(ModelInput input)
    {
        if (!_isLoaded)
            throw new InvalidOperationException("Model not loaded");

        Dictionary<string, float> output = await _jsRuntime.InvokeAsync<Dictionary<string, float>>(_onnxInteropJsFunc, [
            input.Data,
            input.DataShape,
        ]);

        return new Logit
        {
            Normal = output["0"],
            Relax = output["1"],
        };
    }

    public async Task UnloadAsync()
    {
        if (!_isLoaded) 
            throw new InvalidOperationException("Model not loaded");

        await _jsRuntime.InvokeVoidAsync(_onnxUnloadJsFunc);
        _isLoaded = false;
    }
}
