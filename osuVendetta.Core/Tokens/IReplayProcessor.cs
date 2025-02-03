using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.Replays.Data;

namespace osuVendetta.Core.Replays;

public interface IReplayProcessor
{
    AntiCheatModelConfig AntiCheatModelConfig { get; }

    /// <summary>
    /// Checks if a replay is valid to be processed
    /// </summary>
    /// <param name="replayData">Stream with replay data (.osr)</param>
    ReplayValidationResult IsValidReplay(Stream replayData);

    /// <summary>
    /// Creates tokens from a replay
    /// </summary>
    /// <param name="replayData">Stream with replay data (.osr)</param>
    /// <returns></returns>
    ReplayTokens CreateTokensParallel(Stream replayData);
}
