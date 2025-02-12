using osuVendetta.Core.Replays;
using System.Collections;

namespace osuVendetta.Core.IO.Dataset;

public class ReplayDatasetProvider : IReplayDatasetProvider
{
    public int TotalReplays => _datasetArchive.Count;
    public DatasetScalerValues DatasetScalerValues => _datasetArchive.ScalerValues;

    readonly DatasetArchive _datasetArchive;
    readonly bool _shuffle;

    public ReplayDatasetProvider(DatasetArchive datasetArchive, bool shuffle)
    {
        _datasetArchive = datasetArchive;
        _shuffle = shuffle;
    }

    public IEnumerator GetEnumerator()
    {
        return new ReplayDatasetEnumerator(_datasetArchive, _shuffle);
    }
}
