using MudBlazor;
using OsuParsers.Replays;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using osuVendetta.CoreLib.AntiCheat.Data;
using System.Text;

namespace osuVendetta.Web.Client.Services;

public interface IAntiCheatService
{
    Task<BaseAntiCheatResult> ProcessReplayAsync(Replay replay, bool runInParallel);
}

public class AntiCheatService : IAntiCheatService
{
    readonly IAntiCheatConfigProvider _configProvider;
    readonly IAntiCheatModel _antiCheatModel;
    
    AntiCheatRunner? _antiCheatRunner;

    public AntiCheatService(IAntiCheatConfigProvider configProvider, 
        IAntiCheatModel antiCheatModel)
    {
        _configProvider = configProvider;
        _antiCheatModel = antiCheatModel;
    }

    public async Task<BaseAntiCheatResult> ProcessReplayAsync(Replay replay, bool runInParallel)
    {
        await EnsureAntiCheatRunner(); 
        return await _antiCheatRunner!.ProcessReplayAsync(replay, runInParallel);
    }

    async Task EnsureAntiCheatRunner()
    {
        if (_antiCheatRunner is not null)
            return;

        AntiCheatConfig config = await _configProvider.GetConfig()
            ?? throw new NullReferenceException("Config not found");

        string modelPath = await _configProvider.GetModelPath()
            ?? throw new NullReferenceException("Model path not found");

        await _antiCheatModel.LoadAsync(new AntiCheatModelLoadArgs
        {
            AntiCheatConfig = config,
            ModelBytes = Encoding.UTF8.GetBytes(modelPath)
        });

        _antiCheatRunner = new AntiCheatRunner(_antiCheatModel, config);
    }
}
