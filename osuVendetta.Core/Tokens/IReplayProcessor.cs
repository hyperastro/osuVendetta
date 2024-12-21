using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.Replays.Data;

namespace osuVendetta.Core.Replays;

public interface IReplayProcessor
{
    /// <summary>
    /// Checks if a replay is valid to be processed
    /// </summary>
    public ReplayValidationResult IsValidReplay(Replay replay);

    public Task<ReplayTokens> CreateTokensFromFramesAsync(string replayName, List<ReplayFrame> frames, bool runInParallel);
}
