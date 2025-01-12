using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.IO;

public interface IReplayDatasetProvider : IEnumerable
{
    int TotalReplays { get; }
    int GetTotalReplays(ReplayDatasetClass @class);
}

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
            _datasets.Add(datasets[i].Class, datasets[i]);

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

public class ReplayDatasetEntry
{
    public required ReplayDatasetClass Class { get; init; }
    public required ReplayTokens ReplayTokens { get; init; }
}

/// <summary>
/// Replay class
/// </summary>
public enum ReplayDatasetClass
{
    Normal = 0,
    Relax = 1,
}

/// <summary>
/// Represents a dataset made out of replays
/// </summary>
public class ReplayDatasetInfo
{
    /// <summary>
    /// Replay class
    /// </summary>
    public ReplayDatasetClass Class { get; }
    /// <summary>
    /// Entries of the dataset
    /// </summary>
    public List<FileInfo> Entries { get; }

    public ReplayDatasetInfo(ReplayDatasetClass @class, List<FileInfo> entries)
    {
        Entries = entries;
        Class = @class;
    }
}

public class ReplayDatasetEnumerator : IEnumerator
{
    public object Current { get; private set; }

    readonly Dictionary<ReplayDatasetClass, ReplayDatasetInfo> _datasets;
    readonly IReplayProcessor _replayProcessor;

    ReplayDatasetClass _currentClass;
    int _currentIndex;

    public ReplayDatasetEnumerator(IReplayProcessor replayProcessor, Dictionary<ReplayDatasetClass, ReplayDatasetInfo> datasets)
    {
        _datasets = datasets;
        _replayProcessor = replayProcessor;
        Current = LoadEntry(_datasets[_currentClass].Entries[0], _currentClass);
    }

    public bool MoveNext()
    {
        _currentIndex++;

        // reached end
        if (_currentIndex >= _datasets[_currentClass].Entries.Count)
        {
            ReplayDatasetClass nextClass = (ReplayDatasetClass)(((int)_currentClass) + 1);

            if (!_datasets.ContainsKey(nextClass))
                return false;

            _currentClass = nextClass;
            _currentIndex = 0;
        }

        Current = LoadEntry(_datasets[_currentClass].Entries[_currentIndex], _currentClass);
        return true;
    }

    public void Reset()
    {
        _currentClass = default;
        _currentIndex = 0;
    }

    ReplayDatasetEntry LoadEntry(FileInfo entry, ReplayDatasetClass @class)
    {
        using FileStream replayStream = entry.OpenRead();
        ReplayTokens tokens = _replayProcessor.CreateTokensParallel(replayStream);

        return new ReplayDatasetEntry
        {
            Class = @class,
            ReplayTokens = tokens
        };
    }
}