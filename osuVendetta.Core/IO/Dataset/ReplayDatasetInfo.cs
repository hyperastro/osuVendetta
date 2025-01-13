namespace osuVendetta.Core.IO.Dataset;

/// <summary>
/// Represents a dataset made out of replays
/// </summary>
public class ReplayDatasetInfo
{
    /// <summary>
    /// Replay class
    /// <para>This is only used for training</para>
    /// </summary>
    public ReplayDatasetClass? Class { get; }
    /// <summary>
    /// Entries of the dataset
    /// </summary>
    public List<FileInfo> Entries { get; }

    public ReplayDatasetInfo(List<FileInfo> entries)
    {
        int datasetSize = 10;

        if (entries.Count > datasetSize)
            entries = entries.Take(datasetSize).ToList();

        Entries = entries;
    }

    public ReplayDatasetInfo(ReplayDatasetClass @class, List<FileInfo> entries) : this(entries)
    {
        Class = @class;
    }
}
