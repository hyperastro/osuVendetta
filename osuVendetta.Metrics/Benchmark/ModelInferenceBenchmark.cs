using BenchmarkDotNet.Attributes;
using osuVendetta;
using osuVendetta.CLI;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Configuration;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Metrics.Benchmark;

public class ModelInferenceBenchmark
{
#nullable disable // See Setup()
    AntiCheatModel128x3.AntiCheatModel128x3 _antiCheatModel;
    ReplayProcessor _replayProcessor;
    ReplayTokens _replayTokens;
    CLIConfig _config;
#nullable enable

    [GlobalSetup]
    public void Setup()
    {
        _config = BaseConfig.Load<CLIConfig>();
        _antiCheatModel = new AntiCheatModel128x3.AntiCheatModel128x3();
        _antiCheatModel.SetDevice(TorchSharp.DeviceType.CUDA);

        using FileStream modelStream = File.OpenRead(_config.ModelPath);
        using BinaryReader reader = new BinaryReader(modelStream);
        _antiCheatModel.Load(reader);

        _replayProcessor = new ReplayProcessor(_antiCheatModel);

        string replayPath = Path.Combine(_config.GlobalBenchmarkDir, "replay.osr");
        using FileStream replayStream = File.OpenRead(replayPath);
        _replayTokens = _replayProcessor.CreateTokensParallel(replayStream);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _antiCheatModel.Dispose();
    }

    [Benchmark]
    public AntiCheatModelResult RunInference()
    {
        return _antiCheatModel.RunInference(_replayTokens);
    }
}
