using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;
using osuVendetta.Core.IO.Dataset;
using TorchSharp;
using osuVendetta.Core.Training;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Training.Optimizers;

namespace osuVendetta.AntiCheatModel128x3;

public class AntiCheatTrainer : DatasetTrainer
{
    public AntiCheatTrainer(IAntiCheatModel antiCheatModel, IReplayDatasetProvider trainingDatasetProvider, 
        IReplayDatasetProvider testDatasetProvider, ITrainingTracker trainingTracker, float maxGradNorm = 1) : 
        base(antiCheatModel, trainingDatasetProvider, testDatasetProvider, trainingTracker, maxGradNorm)
    {
    }

    protected override WeightedLoss<Tensor, Tensor, Tensor> CreateLossCriterion()
    {
        return new BCEWithLogitsLoss();
    }

    protected override OptimizerHelper CreateOptimizer()
    {
        return new AdamW(AntiCheatModel.GetParameters(), lr: 0.002, weight_decay: 1e-3);
        //return new Prodigy(AntiCheatModel.GetParameters(), weightDecay: .1f, decouple: true);
    }

    protected override lr_scheduler.LRScheduler? CreateScheduler()
    {
        return lr_scheduler.ReduceLROnPlateau(Optimizer, mode: "min", factor: 0.5, patience: 5, verbose: true, min_lr: [1e-6]);
        //return lr_scheduler.CosineAnnealingLR(Optimizer, GlobalMaxEpochs);
        //return null;
    }
}