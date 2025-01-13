using osuVendetta.Core.Replays.Data;

namespace osuVendetta.Core.IO.Dataset;

public class ReplayDatasetEntry
{
    public required ReplayDatasetClass Class { get; init; }
    public required ReplayTokens ReplayTokens { get; init; }
}
