using osuVendetta.Core.Replays;
using System.Collections;

namespace osuVendetta.Core.IO.Dataset;

public class ReplayDatasetProvider : IReplayDatasetProvider
{
    public int TotalReplays { get; private set; }

    readonly Dictionary<ReplayDatasetClass, ReplayDatasetInfo> _datasets;
    readonly IReplayProcessor _replayProcessor;

    public ReplayDatasetProvider(IReplayProcessor replayProcessor, List<ReplayDatasetInfo> datasets)
    {
        _datasets = new Dictionary<ReplayDatasetClass, ReplayDatasetInfo>();
        _replayProcessor = replayProcessor;

        for (int i = 0; i < datasets.Count; i++)
        {
            ReplayDatasetClass @class = datasets[i].Class ??
                throw new InvalidOperationException("Dataset class must be present for training");

            _datasets.Add(@class, datasets[i]);
        }

        TotalReplays = datasets.Sum(d => d.Entries.Count);
    }

    public int GetTotalReplays(ReplayDatasetClass @class)
    {
        if (!_datasets.TryGetValue(@class, out ReplayDatasetInfo? info))
            return 0;

        return info.Entries.Count;
    }

    public IEnumerator GetEnumerator()
    {
        return new ReplayDatasetEnumerator(_replayProcessor, _datasets);
    }
}
