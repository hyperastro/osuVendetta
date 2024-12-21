using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using osuVendetta.Core.Replays.Data;
using TorchSharp;
using osuVendetta.Core.Anticheat.Data;

namespace osuVendetta.Core.AntiCheat;

public interface IAntiCheatModel
{
    AntiCheatModelConfig Config { get; }
    DeviceType Device { get; }

    IAntiCheatModel ToDevice(DeviceType device);
    AntiCheatModelResult RunInference(ReplayTokens tokens);
    void Load(Stream modelWeights);
}
