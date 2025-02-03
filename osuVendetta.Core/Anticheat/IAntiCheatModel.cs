using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using osuVendetta.Core.Replays.Data;
using TorchSharp;
using osuVendetta.Core.Anticheat.Data;
using static TorchSharp.torch;
using TorchSharp.Modules;
using osuVendetta.Core.Training.Utility;
using System.Security.Cryptography;

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
    int MaxBatchSize { get; }

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
    /// <inheritdoc cref="RunInference(ReplayTokens)"/>
    /// </summary>
    /// <param name="tokens"></param>
    /// <param name="isTraining">Should run in training mode</param>
    /// <param name="hiddenStates">Optional hidden states for lstm</param>
    /// <returns></returns>
    LstmData RunInference(ReplayTokens tokens, bool isTraining, (Tensor H0, Tensor C0)? hiddenStates = null);
    /// <summary>
    /// Loads the model
    /// </summary>
    /// <param name="modelSafetensors"></param>
    void Load(Stream model);

    /// <summary>
    /// Saves the model
    /// </summary>
    void Save(Stream model);
    void Reset();
        
    IEnumerable<Parameter> GetParameters();
}
