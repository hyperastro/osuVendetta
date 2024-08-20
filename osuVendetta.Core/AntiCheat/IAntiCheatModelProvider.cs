namespace osuvendetta.Core.AntiCheat;

public abstract record class AntiCheatModelProviderArgs();

public interface IAntiCheatModelProvider
{
    Task<IAntiCheatModel> LoadModelAsync(AntiCheatModelProviderArgs args);
}
