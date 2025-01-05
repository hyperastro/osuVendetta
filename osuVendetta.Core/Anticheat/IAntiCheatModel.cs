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
    /// <summary>
    /// Config the model uses
    /// </summary>
    AntiCheatModelConfig Config { get; }
    /// <summary>
    /// Device the model is currently using
    /// </summary>
    DeviceType Device { get; }

    /// <summary>
    /// Sets <see cref="Device"/> and moves the model to it
    /// </summary>
    /// <param name="device">Device to use for the model</param>
    /// <returns></returns>
    void SetDevice(DeviceType device);
    /// <summary>
    /// Runs the models inference
    /// </summary>
    /// <param name="tokens">Replay tokens</param>
    /// <returns></returns>
    AntiCheatModelResult RunInference(ReplayTokens tokens);
    /// <summary>
    /// Loads the model from safetensors
    /// </summary>
    /// <param name="modelSafetensors"></param>
    void Load(Stream modelSafetensors);
}
