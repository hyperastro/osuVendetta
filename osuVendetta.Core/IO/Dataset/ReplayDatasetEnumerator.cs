using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System.Collections;
using TorchSharp.Modules;

namespace osuVendetta.Core.IO.Dataset;

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
            ReplayDatasetClass nextClass = (ReplayDatasetClass)((int)_currentClass + 1);

            if (!_datasets.ContainsKey(nextClass))
                return false;

            _currentClass = nextClass;
            _currentIndex = 0;
        }
        
        object? entry = LoadEntry(_datasets[_currentClass].Entries[_currentIndex], _currentClass);

        if (entry is null)
            return MoveNext();

        Current = LoadEntry(_datasets[_currentClass].Entries[_currentIndex], _currentClass)!;
        return true;
    }

    public void Reset()
    {
        _currentClass = default;
        _currentIndex = 0;
    }

    ReplayDatasetEntry? LoadEntry(FileInfo entry, ReplayDatasetClass @class)
    {
        ReplayTokens tokens;

        if (entry.Extension.Equals(".txt", StringComparison.CurrentCultureIgnoreCase))
        {
            string[] lines = File.ReadAllLines(entry.FullName);
            List<float> inputs = new List<float>();

            foreach (string line in lines)
            {
                string[] cells = line.Split(',');

                for (int i = 0; i < 6; i++)
                {
                    if (i < cells.Length)
                    {
                        if (float.TryParse(cells[i], out float v))
                        {
                            if (float.IsNaN(v))
                                return null;

                            inputs.Add(v);
                        }
                        else
                        {
                            switch (cells[i].ToLower())
                            {
                                case "m1":
                                    inputs.Add(0);
                                    break;

                                case "m1m2":
                                    inputs.Add(1);
                                    break;

                                case "m2":
                                    inputs.Add(2);
                                    break;

                                default:
                                case "none":
                                    inputs.Add(3);
                                    break;
                            }
                        }
                    }
                    else
                    {
                        inputs.Add(0);
                    }
                }

                foreach (string cell in cells)
                {
                }
            }

            int newInputSize = (int)Math.Ceiling(inputs.Count / 6000.0) * 6000;
            float[] data = inputs.ToArray();

            if (inputs.Count != newInputSize)
                Array.Resize(ref data, newInputSize);

            tokens = new ReplayTokens
            {
                Tokens = data
            };
        }
        else
        {
            using FileStream replayStream = entry.OpenRead();
            tokens = _replayProcessor.CreateTokensParallel(replayStream);
        }

        return new ReplayDatasetEntry
        {
            Class = @class,
            ReplayTokens = tokens
        };
    }
}