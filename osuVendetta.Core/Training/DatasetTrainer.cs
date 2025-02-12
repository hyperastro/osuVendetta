using OsuParsers.Replays;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO.Dataset;
using osuVendetta.Core.Replays.Data;
using osuVendetta.Core.Training.Utility;
using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;


namespace osuVendetta.Core.Training;

/// <summary>
/// <inheritdoc cref="IDatasetTrainer"/>
/// </summary>
public abstract class DatasetTrainer : IDatasetTrainer
{
    record class StepArgs(
        ReplayDatasetEntry Replay);

    record class InferenceResult(
        LstmData Output,
        Tensor Loss);

    record class ReplayTestResult(
        float AverageProbability,
        float[] Probabilities,
        int TruePositives,
        int TrueNegatives,
        int FalsePositives,
        int FalseNegatives);

    /// <summary>
    /// Epoch training hard limit
    /// </summary>
    public static readonly int GlobalMaxEpochs = 9999;

    public LRScheduler? Scheduler { get; init; }
    public WeightedLoss<Tensor, Tensor, Tensor> LossCriterion { get; init; }
    public OptimizerHelper Optimizer { get; init; }
    public IAntiCheatModel AntiCheatModel { get; init; }
    public IReplayDatasetProvider TrainingDatasetProvider { get; init; }
    public IReplayDatasetProvider TestDatasetProvider { get; init; }
    public ITrainingTracker TrainingTracker { get; init; }

    public float MaxGradNorm { get; init; }

    int _largestSegmentSize;
    int _largestTokenSize;
    int _largestReplayTokenSize;

    DateTime _startTime;
    int _totalSegmentSteps;
    int _totalReplays;

    protected DatasetTrainer(IAntiCheatModel antiCheatModel, IReplayDatasetProvider trainingDatasetProvider,
        IReplayDatasetProvider testDatasetProvider, ITrainingTracker trainingTracker,
        float maxGradNorm = 1f)
    {
        AntiCheatModel = antiCheatModel;
        TrainingDatasetProvider = trainingDatasetProvider;
        TestDatasetProvider = testDatasetProvider;
        TrainingTracker = trainingTracker;
        MaxGradNorm = maxGradNorm;

        Optimizer = CreateOptimizer();
        LossCriterion = CreateLossCriterion();
        Scheduler = CreateScheduler();
    }

    public void Train(int epochsToTrain, CancellationToken cancellationToken)
    {
        AntiCheatModel.Reset();
        AntiCheatModel.Config.StandardMean = TrainingDatasetProvider.DatasetScalerValues.StandardMean;
        AntiCheatModel.Config.StandardDeviation = TrainingDatasetProvider.DatasetScalerValues.StandardDeviation;

        int epoch = 0;

        while (!cancellationToken.IsCancellationRequested &&
                epoch < GlobalMaxEpochs &&
                epoch < epochsToTrain)
        {
            EpochStats stats = TrainEpoch(epoch, cancellationToken);
            epoch++;

            using FileStream modelStream = File.Create($"model.{DateTime.Now:dd:MM}.bin");
            AntiCheatModel.Save(modelStream);
        }
    }

    EpochStats TrainEpoch(int epoch, CancellationToken cancellationToken)
    {
        int replayCounter = 0;

        TrainingTracker.SetTrainingState(TrainingState.Training);

        _startTime = DateTime.Now;
        _totalSegmentSteps = 0;
        _totalReplays = 0;

        foreach (ReplayDatasetEntry replay in TrainingDatasetProvider)
        {
            TrainingTracker.SetProgress(replayCounter++, TrainingDatasetProvider.TotalReplays);

            if (replay.ReplayTokens.Tokens.Length == 0)
                continue;

            if (_largestReplayTokenSize <  replay.ReplayTokens.Tokens.Length)
                _largestReplayTokenSize = replay.ReplayTokens.Tokens.Length;

            TrainStep(new StepArgs(replay));
            _totalReplays++;

            TimeSpan diff = DateTime.Now - _startTime;
            double stepsPerSecond = _totalSegmentSteps / diff.TotalSeconds;
            double replaysPerSecond = _totalReplays / diff.TotalSeconds;

            TrainingTracker.SetFooter(
$"Max batch size: {AntiCheatModel.MaxBatchSize}\n" +
//$"Largest replay token size: {_largestReplayTokenSize} (KBytes: {_largestReplayTokenSize * sizeof(float) / 1000}, in batches: {(int)Math.Ceiling(_largestReplayTokenSize / (double)AntiCheatModel.Config.TotalFeatureSizePerChunk)})\n" +
//$"Largest token size processed: {_largestTokenSize} (KBytes: {_largestTokenSize * sizeof(float) / 1000}, in batches: {(int)Math.Ceiling(_largestTokenSize / (double)AntiCheatModel.Config.TotalFeatureSizePerChunk)})\n" +
//$"Largest output size: {_largestSegmentSize}\n" +
$"Steps per second: {stepsPerSecond:n1}\n" +
$"Replays per second: {replaysPerSecond:n1}");
        }

        TrainingTracker.SetTrainingState(TrainingState.Testing);

        EpochStats stats = TestEpoch(epoch, cancellationToken);
        TrainingTracker.SubmitEpoch(stats);

        Scheduler?.step(stats.F1Score, epoch);

        return stats;
    }

    EpochStats TestEpoch(int epoch, CancellationToken cancellationToken)
    {
        //using IDisposable noGrad = torch.no_grad();

        float averageProbability = 0;

        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegativs = 0;

        int replayCounter = 0;
        int invalidCounter = 0;

        foreach (ReplayDatasetEntry replay in TestDatasetProvider)
        {
            TrainingTracker.SetProgress(replayCounter++, TrainingDatasetProvider.TotalReplays - invalidCounter);

            if (replay.ReplayTokens.Tokens.Length == 0)
            {
                invalidCounter++;
                continue;
            }

            ReplayTestResult result = TestReplay(replay);

            truePositives += result.TruePositives;
            trueNegatives += result.TrueNegatives;
            falsePositives += result.FalsePositives;
            falseNegativs += result.FalseNegatives;

            averageProbability += result.AverageProbability;
        }

        averageProbability /= TestDatasetProvider.TotalReplays - invalidCounter;

        float f1Score = F1Score.CalculateF1Score(truePositives, falsePositives, falseNegativs);

        return new EpochStats(epoch, f1Score, truePositives, trueNegatives, 
            falsePositives, falseNegativs, averageProbability);
    }

    ReplayTestResult TestReplay(ReplayDatasetEntry replay)
    {
        using IDisposable scope = NewDisposeScope();

        LstmData data = AntiCheatModel.RunInference(replay.ReplayTokens, false);
        float[] probabilities = sigmoid(data.Data).ToArray<float>();

        float averageProbability = 0;
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            bool isPositivePrediction = probabilities[i] >= 0.5f;
            bool isPositiveLabel = replay.Class == ReplayDatasetClass.Relax;

            if (isPositivePrediction)
            {
                if (isPositiveLabel)
                    truePositives++;
                else
                    falsePositives++;
            }
            else
            {
                if (isPositiveLabel)
                    falseNegatives++;
                else
                    trueNegatives++;
            }


            averageProbability += probabilities[i];
        }

        averageProbability /= probabilities.Length;

        _totalSegmentSteps++;

        return new ReplayTestResult(averageProbability, probabilities,
            truePositives, trueNegatives, falsePositives, falseNegatives);
    }

    void TrainStep(StepArgs args)
    {
        using IDisposable scope = NewDisposeScope();

        Optimizer.zero_grad();

        int maxFeatureSize = AntiCheatModel.MaxBatchSize * AntiCheatModel.Config.TotalFeatureSizePerChunk;

        if (args.Replay.ReplayTokens.Tokens.Length > maxFeatureSize)
        {
            Span<float> tokensRemaining = new Span<float>(args.Replay.ReplayTokens.Tokens);

            (Tensor H0, Tensor C0)? lastData = null;

            while (tokensRemaining.Length > 0)
            {
                int toTake = Math.Min(tokensRemaining.Length, maxFeatureSize);

                Span<float> tokens = tokensRemaining[..toTake];
                tokensRemaining = tokensRemaining[toTake..];

                float[] currentTokens = new float[maxFeatureSize];
                tokens.CopyTo(currentTokens);

                ReplayTokens replayTokens = new ReplayTokens
                {
                    Tokens = currentTokens
                };

                (Tensor H0, Tensor C0) data = RunTrainingInference(replayTokens, args.Replay.Class, lastData);

                lastData?.H0.Dispose();
                lastData?.C0.Dispose();
                lastData = data;
            }
        }
        else
        {
           _ = RunTrainingInference(args.Replay.ReplayTokens, args.Replay.Class);
        }

        nn.utils.clip_grad_norm_(AntiCheatModel.GetParameters(), MaxGradNorm);
        _ = Optimizer.step();
    }

    (Tensor H0, Tensor C0) RunTrainingInference(ReplayTokens tokens, ReplayDatasetClass @class, (Tensor H0, Tensor C0)? hiddenStates = null)
    {
        if (tokens.Tokens.Length > _largestTokenSize)
            _largestTokenSize = tokens.Tokens.Length;

        using IDisposable scope = NewDisposeScope();

        LstmData data = AntiCheatModel.RunInference(tokens, true, hiddenStates);
        Tensor classLabels = CreateClassLabels((int)data.Data.shape[0], @class);
        Tensor loss = LossCriterion.forward(data.Data, classLabels);

        loss.backward();

        if (data.Data.shape[0] > _largestSegmentSize)
            _largestSegmentSize = (int)data.Data.shape[0];

        _totalSegmentSteps++;

        (Tensor H0, Tensor C0) result = data.HiddenState!.Value;
        result.H0.MoveToOuterDisposeScope();
        result.C0.MoveToOuterDisposeScope();

        return result;
    }

    Tensor CreateClassLabels(int length, ReplayDatasetClass @class)
    {
        float[] data = new float[length];
        Array.Fill(data, (int)@class);

        return tensor(data);
    }

    /// <summary>
    /// Creates the scheduler
    /// </summary>
    /// <returns>Scheduler</returns>
    protected abstract LRScheduler? CreateScheduler();
    /// <summary>
    /// Creates the loss criterion
    /// </summary>
    /// <returns>Loss criterion</returns>
    protected abstract WeightedLoss<Tensor, Tensor, Tensor> CreateLossCriterion();
    /// <summary>
    /// Creates the optimizer
    /// </summary>
    /// <returns>Optimizer</returns>
    protected abstract OptimizerHelper CreateOptimizer();
} 
