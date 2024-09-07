using osuVendetta.Core.AntiCheat;

namespace osuVendetta.CLI.AntiCheat;

public class AntiCheatModelProvider : IAntiCheatModelProvider
{
    readonly string _modelPath = @$"192x2.onnx";

    public async Task<IAntiCheatModel> LoadModelAsync(AntiCheatModelProviderArgs args)
    {
        AntiCheatModel192x2 model = new AntiCheatModel192x2(_modelPath);
        await model.LoadAsync();

        return model;
    }
}
