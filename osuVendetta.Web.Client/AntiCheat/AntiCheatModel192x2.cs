using Microsoft.JSInterop;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;

namespace osuVendetta.Web.Client.AntiCheat;

public class AntiCheatModel192x2 : AntiCheatModel
{
    readonly static string _onnxInteropJsFunc = "antiCheat.run";
    readonly IJSRuntime _jsRuntime;

    public AntiCheatModel192x2(IJSRuntime jsRuntime) : base(string.Empty)
    {
        _jsRuntime = jsRuntime;
    }

    static MaxValueIndex GetMaxValueIndex(Dictionary<string, float> values)
    {
        string maxIndex = string.Empty;
        float maxValue = float.MinValue;

        foreach (KeyValuePair<string, float> kvp in values)
        {
            if (kvp.Value > maxValue)
            {
                maxIndex = kvp.Key;
                maxValue = kvp.Value;
            }
        }

        return new MaxValueIndex(int.Parse(maxIndex), maxValue);
    }

    public override async Task LoadAsync()
    {

    }

    protected override void Unload()
    {
        // don't need to unload js
    }

    protected override async Task<ProbabilityResult> RunModelAsync(Memory<float> input, long[] shape)
    {
        Dictionary<string, float> output = await _jsRuntime.InvokeAsync<Dictionary<string, float>>(_onnxInteropJsFunc, new object[]
        {
            input,
            shape
        });

        return new ProbabilityResult(output["0"], output["1"]);
    }
}
