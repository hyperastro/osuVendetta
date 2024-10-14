using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using System;
using System.Text.Json;
using System.Threading.Tasks;

namespace osuVendetta.Wasm;

public class AntiCheatModel192x2 : IAntiCheatModel
{
    static readonly string[] _inputNames = ["input"];

    public Task LoadAsync(AntiCheatModelLoadArgs loadArgs)
    {
        ArgumentNullException.ThrowIfNull(Core.ModelPath);

        Core.LoadModel(Core.ModelPath);

        return Task.CompletedTask;
    }

    public Logit Run(ModelInput input)
    {
        double[] data = new double[input.Data.Length];

        for (int i = 0; i < data.Length; i++)
            data[i] = input.Data.Span[i];

        int[] dataShape = new int[input.DataShape.Length];

        for (int i = 0; i < dataShape.Length; i++)
            dataShape[i] = (int)input.DataShape[i];

        return JsonSerializer.Deserialize<Logit>(Core.RunModel(_inputNames, data, dataShape), new JsonSerializerOptions(JsonSerializerDefaults.Web));
    }

    public Task UnloadAsync()
    {
        throw new System.NotImplementedException();
    }
}
