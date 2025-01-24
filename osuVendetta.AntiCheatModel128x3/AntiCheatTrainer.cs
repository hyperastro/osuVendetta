using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;
using osuVendetta.Core.Utility;
using osuVendetta.Core.Optimizers;
using osuVendetta.Core.IO.Dataset;
using TorchSharp;
using System.Numerics;
using osuVendetta.Core.F1;

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
        float lastF1,
        EpochTrainingParameters Parameters,
        IReplayDatasetProvider DatasetProvider,
        IProgressReporter ProgressReporter);

    readonly static int _epochTrainingLimit = 9999;
    readonly static int _epochTrainingWastedStop = 20;
    readonly static int _epochTrainingMin = 5;

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
        return new AdamW(_model.parameters(), lr: 0.002, weight_decay: 1e-3);
        //return new Prodigy(_model.parameters(), weightDecay: .1f, decouple: true);
    }

    WeightedLoss<Tensor, Tensor, Tensor> CreateLossCriterion()
    {
        return new BCEWithLogitsLoss();
    }

    lr_scheduler.LRScheduler? CreateScheduler(OptimizerHelper optimizer)
    {
        //return lr_scheduler.CosineAnnealingLR(optimizer, _epochTrainingLimit);
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode: "min", factor: 0.5, patience: 5, verbose: true, min_lr: [1e-6]);
        return null;
    }


    void PrepareModel()
    {
        PrepareSeed();

        autograd.set_detect_anomaly(true);

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
        float bestEpochF1 = 0;

        float lastF1 = 0;

        int epochsWithoutImprovement = 0;
        while (epochsWithoutImprovement < _epochTrainingWastedStop &&
               currentEpoch < _epochTrainingLimit)
        {
            EpochParameters epochParams = new EpochParameters(
                currentEpoch, maxGradNorm, avgAccuracy,
                avgLoss, lastF1, parameters, tokenProvider, progress);

            (float runningLoss, float runningAccuracy, float f1Score) = TrainEpoch(epochParams);

            totalAccuracy += runningAccuracy;
            totalLoss += runningLoss;

            // use first 2 epochs for warmup
            if (currentEpoch > 2)
            {
                if (f1Score < bestEpochF1)
                {
                    // train for atleast X epochs before starting to count for bad epochs
                    if (currentEpoch >= _epochTrainingMin)
                        epochsWithoutImprovement++;

                    // return to last state that had any improvement
                    //_model.load_state_dict(bestState);
                }
                else
                {
                    epochsWithoutImprovement = 0;
                    bestState = _model.state_dict();
                    bestEpoch = currentEpoch;
                    bestEpochF1 = f1Score;
                }
            }

            currentEpoch++;

            lastF1 = f1Score;
            avgAccuracy = totalAccuracy / currentEpoch;
            avgLoss = totalLoss / currentEpoch;
        }

        _model.load_state_dict(bestState);
    }

    (float runningLoss, float runningAccuracy, float f1Score) TrainEpoch(EpochParameters parameters)
    {
        float totalLoss = 0;
        float avgLoss = 0;
        float epochRunningF1Score = 0;

        float totalAccuracy = 0;
        float avgAccuracy = 0;

        int totalTruePositives = 0;
        int totalFalsePositives = 0;
        int totalFalseNegatives = 0;

        parameters.ProgressReporter.SetMaxProgress(parameters.DatasetProvider.TotalReplays);

        int replayCounter = 0;
        foreach (ReplayDatasetEntry entry in parameters.DatasetProvider)
        {
            parameters.ProgressReporter.SetProgressTitle(
$"Epoch {parameters.Epoch}: {replayCounter} / {parameters.DatasetProvider.TotalReplays} " +
$"(Current Acc/Loss/F1: {avgAccuracy:n4}/{avgLoss:n4}/{epochRunningF1Score:n4}, Last Epoch F1: {parameters.lastF1}) ");

            parameters.ProgressReporter.Increment();

            using (IDisposable disposeScope = NewDisposeScope())
            {
                (float runningLoss, float runningAccuracy, int truePositives, int falsePositives, int falseNegatives) = Step(parameters, entry);

                totalLoss += runningLoss;
                totalAccuracy += runningAccuracy;
                totalTruePositives += truePositives;
                totalFalsePositives += falsePositives;
                totalFalseNegatives += falseNegatives;

                replayCounter++;

                avgLoss = totalLoss / replayCounter;
                avgAccuracy = totalAccuracy / replayCounter;
                epochRunningF1Score = F1Score.CalculateF1Score(totalTruePositives, totalFalsePositives, totalFalseNegatives);
            }
        }

        _model.train();
        parameters.Parameters.Scheduler?.step(avgLoss, parameters.Epoch);
        
        float f1Score = F1Score.CalculateF1Score(totalTruePositives, totalFalsePositives, totalFalseNegatives);

        return (avgLoss, avgAccuracy, f1Score);
    }

    (float runningLoss, float runningAccuracy, int truePositives, int falsePositives, int falseNegatives) Step(EpochParameters parameters, ReplayDatasetEntry entry)
    {
        parameters.Parameters.Optimizer.zero_grad();

        float[] labelsArray = new float[(int)Math.Ceiling(entry.ReplayTokens.Tokens.Length / (double)_model.Config.TotalFeatureSizePerChunk)];
        for (int i = 0; i < labelsArray.Length; i++)
            labelsArray[i] = (int)entry.Class;

        Tensor labels = tensor(labelsArray);

        LstmData data = _model.RunInference(entry.ReplayTokens, true);

        Tensor loss = parameters.Parameters.LossCriterion.forward(data.Data, labels);

        loss.backward();

        //foreach (Parameter parameter in _model.parameters())
        //    Console.WriteLine($"Parameter '{parameter.name}' requires grad: {parameter.requires_grad} (is null: {parameter.grad is null}");

        nn.utils.clip_grad_norm_(_model.parameters(), parameters.MaxGradNorm);
        _ = parameters.Parameters.Optimizer.step();

        //foreach (Parameter parameter in _model.parameters())
        //    Console.WriteLine($"Parameter '{parameter.name}' requires grad: {parameter.requires_grad} (is null: {parameter.grad is null}");

        float runningLoss = loss.item<float>();
        float runningAccuracy = LogitsToAccuracy(data.Data, entry.Class);

        // Calculate F1 score directly
        float[] probabilities = sigmoid(data.Data.ToArray<float>()).ToArray<float>(); //Convert Logits into Probablilties
        int truePositives = 0, falsePositives = 0, falseNegatives = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            bool isPositivePrediction = probabilities[i] >= 0.5f;
            bool isPositiveLabel = labelsArray[i] == 1;

            if (isPositivePrediction && isPositiveLabel)
            {
                truePositives++;
            }
            else if (isPositivePrediction && !isPositiveLabel)
            {
                falsePositives++;
            }
            else if (!isPositivePrediction && isPositiveLabel)
            {
                falseNegatives++;
            }
        }

        return (runningLoss, runningAccuracy, truePositives, falsePositives, falseNegatives);
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
