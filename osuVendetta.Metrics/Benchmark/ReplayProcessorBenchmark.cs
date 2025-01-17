using BenchmarkDotNet.Attributes;
using osuVendetta;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Metrics.Benchmark;

public class ReplayProcessorBenchmark
{
#nullable disable // See Setup()
    AntiCheatModel128x3.AntiCheatModel128x3 _antiCheatModel;
    ReplayProcessor _replayProcessor;
    FileStream _replayStream;
#nullable enable

    [GlobalSetup]
    public void Setup()
    {
        _antiCheatModel = new AntiCheatModel128x3.AntiCheatModel128x3();
        _replayProcessor = new ReplayProcessor(_antiCheatModel);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        _replayStream?.Dispose();
        _replayStream = File.OpenRead("Data/Benchmark/replay.osr");
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _replayStream.Dispose();
    }

    [Benchmark]
    public ReplayValidationResult ValidateReplay()
    {
        return _replayProcessor.IsValidReplay(_replayStream);
    }

    [Benchmark]
    public ReplayTokens CreateTokensParallel()
    {
        return _replayProcessor.CreateTokensParallel(_replayStream);
    }
}
