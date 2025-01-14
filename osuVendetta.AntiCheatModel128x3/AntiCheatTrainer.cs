using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;
using osuVendetta.Core.Utility;
using osuVendetta.Core.Optimizers;
using osuVendetta.Core.IO.Dataset;
using TorchSharp;
using System.Numerics;

namespace osuVendetta.AntiCheatModel128x3;

public class EpochTrainingParameters : IDisposable
{
    public bool IsDisposed { get; private set; }

    public OptimizerHelper Optimizer { get; }
    public WeightedLoss<Tensor, Tensor, Tensor> LossCriterion { get; }
    public lr_scheduler.LRScheduler? Scheduler { get; set; }

    public EpochTrainingParameters(OptimizerHelper optimizer, 
        WeightedLoss<Tensor, Tensor, Tensor> lossCriterion)
    {
        Optimizer = optimizer;
        LossCriterion = lossCriterion;
    }

    public void Dispose()
    {
        if (IsDisposed)
            return;

        Optimizer.Dispose();
        LossCriterion.Dispose();

        IsDisposed = true;
    }
}

public class AntiCheatTrainer
{
    record class EpochParameters(
        int Epoch,
        float MaxGradNorm,
        float avgAccuracy,
        float avgLoss,
        EpochTrainingParameters Parameters,
        IReplayDatasetProvider DatasetProvider,
        IProgressReporter ProgressReporter);

    readonly static int _epochTrainingLimit = 9999;
    readonly static int _epochTrainingWastedStop = 20;
    readonly static int _epochTrainingMin = 100;

    readonly AntiCheatModel128x3 _model;

    public AntiCheatTrainer(AntiCheatModel128x3 model)
    {
        _model = model;
    }

    public void RunTraining(IReplayDatasetProvider tokenProvider, IProgressReporter progressReporter)
    {
        // implement hyperparameter search
        // implement progress reports

        PrepareModel();

        using EpochTrainingParameters parameters = new EpochTrainingParameters(
            CreateOptimizer(),
            CreateLossCriterion());

        parameters.Scheduler = CreateScheduler(parameters.Optimizer);

        TrainEpochs(parameters, tokenProvider, progressReporter);
    }

    OptimizerHelper CreateOptimizer()
    {
        //return new AdamW(_model.parameters(), lr: 0.002, weight_decay: 1e-3);
        return new Prodigy(_model.parameters(), weightDecay: .1f, decouple: true);
    }

    WeightedLoss<Tensor, Tensor, Tensor> CreateLossCriterion()
    {
        return new BCEWithLogitsLoss();
    }

    lr_scheduler.LRScheduler? CreateScheduler(OptimizerHelper optimizer)
    {
        //return lr_scheduler.CosineAnnealingLR(optimizer, _epochTrainingLimit);
        //return lr_scheduler.ReduceLROnPlateau(optimizer, mode: "min", factor: 0.5, patience: 5, verbose: true, min_lr: [1e-6]);
        return null;
    }


    void PrepareModel()
    {
        PrepareSeed();

        autograd.set_detect_anomaly(true);
        set_default_device(torch.device(DeviceType.CUDA));

        _model.SetDevice(DeviceType.CUDA);

        if (!_model.training)
            _model.train();
    }

    void PrepareSeed()
    {
        const long SEED = 123456;

        random.manual_seed(SEED);
        manual_seed(SEED);
        cuda.manual_seed(SEED);
        cuda.manual_seed_all(SEED);
    }

    void TrainEpochs(EpochTrainingParameters parameters, IReplayDatasetProvider tokenProvider, IProgressReporter progress)
    {
        float maxGradNorm = 1;

        Dictionary<string, Tensor> bestState = _model.state_dict();

        float totalAccuracy = 0;
        float avgAccuracy = 0;

        float totalLoss = 0;
        float avgLoss = 0;
        
        int currentEpoch = 0;
        int bestEpoch = 0;
        float bestEpochAccuracy = 0;

        int epochsWithoutImprovement = 0;
        while (epochsWithoutImprovement < _epochTrainingWastedStop &&
               currentEpoch < _epochTrainingLimit)
        {
            EpochParameters epochParams = new EpochParameters(
                currentEpoch, maxGradNorm, avgAccuracy, 
                avgLoss, parameters, tokenProvider, progress);

            (float runningLoss, float runningAccuracy) = TrainEpoch(epochParams);

            totalAccuracy += runningAccuracy;
            totalLoss += runningLoss;

            if (runningAccuracy < bestEpochAccuracy)
            {
                // train for atleast X epochs before starting to count for bad epochs
                if (currentEpoch >= _epochTrainingMin)
                    epochsWithoutImprovement++;
            }
            else
            {
                epochsWithoutImprovement = 0;
                bestState = _model.state_dict();
                bestEpoch = currentEpoch;
                bestEpochAccuracy = runningAccuracy;
            }

            currentEpoch++;
            avgAccuracy = totalAccuracy / currentEpoch;
            avgLoss = totalLoss / currentEpoch;
        }

        _model.load_state_dict(bestState);
    }

    (float runningLoss, float runningAccuracy) TrainEpoch(EpochParameters parameters)
    {
        float totalLoss = 0;
        float avgLoss = 0;

        float totalAccuracy = 0;
        float avgAccuracy = 0;

        parameters.ProgressReporter.SetMaxProgress(parameters.DatasetProvider.TotalReplays);

        int replayCounter = 0;
        foreach (ReplayDatasetEntry entry in parameters.DatasetProvider)
        {
            parameters.ProgressReporter.SetProgressTitle(
$"Epoch {parameters.Epoch} Replay {replayCounter} / {parameters.DatasetProvider.TotalReplays} " +
$"(Avg Accuracy: {avgAccuracy:n8}, Avg Loss: {avgLoss:n8}) ");

            parameters.ProgressReporter.Increment();

            using (IDisposable disposeScope = NewDisposeScope())
            {
                (float runningLoss, float runningAccuracy) = Step(parameters, entry);

                totalLoss += runningLoss;
                totalAccuracy += runningAccuracy;

                replayCounter++;

                avgLoss = totalLoss / replayCounter;
                avgAccuracy = totalAccuracy / replayCounter;
            }
        }

        _model.train();
        parameters.Parameters.Scheduler?.step(avgLoss, parameters.Epoch);

        return (avgLoss, avgAccuracy);
    }

    (float runningLoss, float runningAccuracy) Step(EpochParameters parameters, ReplayDatasetEntry entry)
    {
        parameters.Parameters.Optimizer.zero_grad();

        float[] labelsArray = new float[(int)Math.Ceiling(entry.ReplayTokens.Tokens.Length / (double)_model.Config.TotalFeatureSizePerChunk)];
        for (int i = 0; i < labelsArray.Length; i++)
            labelsArray[i] = (int)entry.Class;

        Tensor labels = tensor(labelsArray);

        LstmData data = _model.RunInference(entry.ReplayTokens, true);
        float[] segments = data.Data.ToArray<float>();
        Tensor segmentOutputs = segments;

        Tensor loss = parameters.Parameters.LossCriterion.forward(segmentOutputs.requires_grad_(), labels.requires_grad_());

        loss.backward();
        nn.utils.clip_grad_norm_(_model.parameters(), parameters.MaxGradNorm);
        _ = parameters.Parameters.Optimizer.step();

        float runningLoss = loss.item<float>();
        float runningAccuracy = LogitsToAccuracy(data.Data, entry.Class);

        return (runningLoss, runningAccuracy);
    }

    float LogitsToAccuracy(Tensor logits, ReplayDatasetClass @class)
    {
        float[] segments = sigmoid(logits).ToArray<float>();
        
        float accuracy = segments.Sum(segment =>
        {
            switch (@class)
            {
                case ReplayDatasetClass.Normal:
                    if (segment < .5f)
                        return 1f;
                    else
                        return 0f;

                case ReplayDatasetClass.Relax:
                    if (segment >= .5f)
                        return 1f;
                    else
                        return 0f;

                default:
                    throw new InvalidOperationException($"Unkown class type {@class}");
            }
        });

        return accuracy / segments.Length;
    }
}
