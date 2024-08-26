using osuVendetta.Core.AntiCheat.Data;

namespace osuVendetta.Core.AntiCheat;

public record class InputArgs(
    float[] InputData,
    long[] Dimensions
);

public interface IAntiCheatModel : IDisposable
{
    Task<AntiCheatResult> RunModelAsync(InputArgs args);
}
