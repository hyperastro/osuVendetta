using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO.Dataset;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;


namespace osuVendetta.Core.Training;

/// <summary>
/// Anticheat model trainer
/// </summary>
public interface IDatasetTrainer
{
    /// <summary>
    /// Scheduler
    /// </summary>
    LRScheduler? Scheduler { get; }
    /// <summary>
    /// Loss criterion
    /// </summary>
    WeightedLoss<Tensor, Tensor, Tensor> LossCriterion { get; }
    /// <summary>
    /// Optimizer
    /// </summary>
    OptimizerHelper Optimizer { get; }
    /// <summary>
    /// Anticheat model
    /// </summary>
    IAntiCheatModel AntiCheatModel { get; }
    /// <summary>
    /// Replay dataset provider
    /// </summary>
    IReplayDatasetProvider TrainingDatasetProvider { get; }
    ITrainingTracker TrainingTracker { get; }
    /// <summary>
    /// Max grad norm for clipping
    /// </summary>
    float MaxGradNorm { get; }

    /// <summary>
    /// Trains the <see cref="AntiCheatModel"/>
    /// </summary>
    /// <param name="epochsToTrain">Amount of epochs to count</param>
    /// <param name="cancellationToken">Cancellation token</param>
    void Train(int epochsToTrain, CancellationToken cancellationToken);
}
