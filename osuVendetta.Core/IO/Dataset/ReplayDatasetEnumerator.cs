using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System.Collections;
using TorchSharp.Modules;

namespace osuVendetta.Core.IO.Dataset;

public class ReplayDatasetEnumerator : IEnumerator
{
    public object? Current { get; private set; }

    int _currentIndex;

    readonly DatasetArchive _datasetArchive;
    readonly bool _shuffle;
    readonly HashSet<int> _usedIndices;

    public ReplayDatasetEnumerator(DatasetArchive datasetArchive, bool shuffle)
    {
        _datasetArchive = datasetArchive;
        _shuffle = shuffle;
        _usedIndices = new HashSet<int>();
    }

    public bool MoveNext()
    {
        if (_shuffle)
        {
            if (_usedIndices.Count == _datasetArchive.Count)
            {
                Reset();
                return false;
            }

            do
            {
                _currentIndex = Random.Shared.Next(0, _datasetArchive.Count);
            }
            while (_usedIndices.Contains(_currentIndex));

            _usedIndices.Add(_currentIndex);
            ReadEntry();

            return true;
        }
        else
        {
            if (_currentIndex == _datasetArchive.Count)
            {
                Reset();
                return false;
            }

            _currentIndex++;
            ReadEntry();

            return true;
        }
    }

    public void Reset()
    {
        _currentIndex = 0;
        _usedIndices.Clear();
    }

    void ReadEntry()
    {
        Current = _datasetArchive[_currentIndex];
    }
}