using osuvendetta.Core.AntiCheat.Data;

namespace osuvendetta.Core.AntiCheat;

public record class InputArgs(
    float[] InputData,
    long[] Dimensions
);

public interface IAntiCheatModel : IDisposable
{
    Task<AntiCheatResult> RunModelAsync(InputArgs args);
}
