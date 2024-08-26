using Microsoft.JSInterop;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;

namespace osuVendetta.Web.Client.AntiCheat
{
    public class AntiCheatModel : IAntiCheatModel
    {
        readonly static string _onnxInteropJsFunc = "antiCheat.run";
        readonly IJSRuntime _jsRuntime;

        public AntiCheatModel(IJSRuntime jsRuntime)
        {
            _jsRuntime = jsRuntime;
        }

        public async Task<AntiCheatResult> RunModelAsync(InputArgs args)
        {
            Dictionary<string, float> output = await _jsRuntime.InvokeAsync<Dictionary<string, float>>(_onnxInteropJsFunc, new object[]
            {
                args.InputData,
                args.Dimensions
            });

            MaxValueIndex maxValueIndex = GetMaxValueIndex(output);

            switch (maxValueIndex.MaxIndex)
            {
                case 0:
                    return AntiCheatResult.Relax();

                case 1:
                    return AntiCheatResult.Normal();

                default:
                    return AntiCheatResult.Invalid($"Unkown result classname index: {maxValueIndex.MaxIndex}");
            }
        }

        public void Dispose()
        {
            // don't need to cleanup js
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
    }
}
