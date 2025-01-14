using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;
using osuVendetta.Core.Utility;
using osuVendetta.Core.Optimizers;
using osuVendetta.Core.IO.Dataset;
using TorchSharp;
using System.Numerics;

namespace osuVendetta.AntiCheatModel128x3;

public class AntiCheatTrainer
{
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

        set_default_device(torch.device(DeviceType.CUDA));
        _model.SetDevice(DeviceType.CUDA);

        if (!_model.training)
            _model.train();

        autograd.set_detect_anomaly(true);

        Dictionary<ReplayDatasetClass, int> totalReplayCounts = new Dictionary<ReplayDatasetClass, int>
        {
            { ReplayDatasetClass.Normal, tokenProvider.GetTotalReplays(ReplayDatasetClass.Normal) },
            { ReplayDatasetClass.Relax, tokenProvider.GetTotalReplays(ReplayDatasetClass.Relax) }
        };

        //using Tensor classWeights = CreateClassWeights(totalReplayCounts);
        //using CrossEntropyLoss criterion = new CrossEntropyLoss();
        using Prodigy optimizer = new Prodigy(_model.parameters(), weightDecay: .1f, decouple: true);
        //lr_scheduler.LRScheduler scheduler = lr_scheduler.CosineAnnealingLR(optimizer, _epochTrainingLimit);

        using BCEWithLogitsLoss criterion = new BCEWithLogitsLoss();
        //using AdamW optimizer = new AdamW(_model.parameters(), lr: 0.002, weight_decay: 1e-3);
        //lr_scheduler.LRScheduler scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode: "min", factor: 0.5, patience: 5, verbose: true, min_lr: [1e-6]);

        TrainEpochs(optimizer, /*scheduler*/null, criterion, tokenProvider, progressReporter);
    }

    void TrainEpochs(OptimizerHelper optimizer, lr_scheduler.LRScheduler? scheduler, BCEWithLogitsLoss criterion,
        IReplayDatasetProvider tokenProvider, IProgressReporter progress)
    {
        float maxGradNorm = 1;

        Dictionary<string, Tensor> bestState = _model.state_dict();
        float bestAccuracy = 0;
        int bestEpoch = 0;
        int currentEpoch = 0;

        float lastAccuracy = 0;
        float lastLoss = 0;

        int epochsWithoutImprovement = 0;
        while (epochsWithoutImprovement < _epochTrainingWastedStop &&
               currentEpoch < _epochTrainingLimit)
        {
            (float loss, float accuracy) = TrainEpoch(currentEpoch, maxGradNorm, optimizer, scheduler, criterion, tokenProvider, progress, lastAccuracy, lastLoss);
            lastAccuracy = accuracy;
            lastLoss = loss;


            //progress.SetProgressTitle($"Epoch {currentEpoch} (Accuracy: {accuracy}, Loss: {loss}, LR: {optimizer.ParamGroups.First().LearningRate}) ");
            //progress.SetMaxProgress(double.MaxValue);
            //progress.SetCurrentProgress(currentEpoch);

            if (accuracy < bestAccuracy)
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
                bestAccuracy = accuracy;
            }

            currentEpoch++;
        }

        _model.load_state_dict(bestState);
    }

    (float loss, float accuracy) TrainEpoch(int epoch, float maxGradNorm, OptimizerHelper optimizer,
        lr_scheduler.LRScheduler? scheduler, BCEWithLogitsLoss criterion, IReplayDatasetProvider tokenProvider,
        IProgressReporter progress, float lastAccuracy, float lastLoss)
    {
        float runningLoss = 0;
        float accuracy = 0;
        float avgLoss = 0;
        float avgAccuracy = 0;

        progress.SetMaxProgress(tokenProvider.TotalReplays);
        //progress.SetProgressTitle($"Replay 0 / {tokenProvider.TotalReplays}");

        int replayCounter = 0;
        foreach (ReplayDatasetEntry entry in tokenProvider)
        {
            progress.SetProgressTitle($"Epoch {epoch} Replay {replayCounter} / {tokenProvider.TotalReplays} (Accuracy: {avgAccuracy:n8}, Loss: {avgLoss:n8}, LR: {optimizer.ParamGroups.First().LearningRate}) ");
            progress.Increment();

            using (IDisposable disposeScope = NewDisposeScope())
            {
                optimizer.zero_grad();

                float[] labelsArray = new float[(int)Math.Ceiling(entry.ReplayTokens.Tokens.Length / (double)_model.Config.TotalFeatureSizePerChunk)];
                for (int i = 0; i < labelsArray.Length; i++)
                    labelsArray[i] = (int)entry.Class;

                using Tensor labels = tensor(labelsArray);

                using LstmData data = _model.RunInference(entry.ReplayTokens, true);
                float[] segments = data.Data.ToArray<float>();
                using Tensor segmentOutputs = segments;

                using Tensor loss = criterion.forward(segmentOutputs.requires_grad_(), labels.requires_grad_());

                loss.backward();
                nn.utils.clip_grad_norm_(_model.parameters(), maxGradNorm);

                runningLoss += loss.item<float>();
                replayCounter++;

                using Tensor segmentsSigmoid = sigmoid(data.Data);
                segments = segmentsSigmoid.ToArray<float>();

                int tempAcc = 0;
                for (int i = 0; i < segments.Length; i++)
                {
                    switch (entry.Class)
                    {
                        case ReplayDatasetClass.Normal:
                            if (segments[i] < .5f)
                                tempAcc++;
                            break;

                        case ReplayDatasetClass.Relax:
                            if (segments[i] >= .5f)
                                tempAcc++;
                            break;
                    }
                }

                tempAcc /= segments.Length;
                accuracy += tempAcc;
            }

            using Tensor optimizerStep = optimizer.step();

            if (replayCounter % 100 == 0)
                GC.Collect();

            avgAccuracy = accuracy / replayCounter;
            avgLoss = runningLoss / replayCounter;
        }

        _model.train();
        scheduler?.step(runningLoss, epoch);
        return (runningLoss, accuracy);
    }

    Tensor CreateClassWeights(Dictionary<ReplayDatasetClass, int> replayCount)
    {
        int[] classCounts = replayCount.OrderBy(kvp => (int)kvp.Key).Select(kvp => kvp.Value).ToArray();
        int totalClassCounts = classCounts.Sum();
        float[] classWeights = classCounts.Select(v => (float)(totalClassCounts / v)).ToArray();

        return tensor(classWeights);
    }



    // AdamW impl
    //public void RunTraining(IReplayTokenProvider tokenProvider, IProgressReporter progressReporter)
    //{
    //    // implement prodigy https://github.com/konstmish/prodigy/blob/main/prodigyopt/prodigy.py
    //    // implement hyperparameter search
    //    // implement progress reports

    //    if (!training)
    //        train();

    //    Dictionary<AntiCheatClass, int> replayCount = tokenProvider.GetReplayCount();

    //    using Tensor labels = CreateTrainingLabels();
    //    using Tensor classWeights = CreateClassWeights(replayCount);
    //    using BCEWithLogitsLoss criterion = new BCEWithLogitsLoss(pos_weights: classWeights);
    //    using AdamW optimizer = new AdamW(parameters(), lr: 0.002, weight_decay: 1e-3);
    //    //using Prodigy optimizer = new Prodigy();

    //    optim.lr_scheduler.LRScheduler scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode: "min", factor: 0.5, patience: 5, verbose: true, min_lr: [1e-6]);

    //    TrainEpochs(optimizer, criterion, tokenProvider, labels);
    //}

    //void TrainEpochs(AdamW optimizer, BCEWithLogitsLoss criterion, 
    //    IReplayTokenProvider tokenProvider, Tensor labels)
    //{
    //    float maxGradNorm = 1;

    //    Dictionary<string, Tensor> bestState = state_dict();
    //    float bestAccuracy = 0;
    //    int bestEpoch = 0;
    //    int currentEpoch = 0;
    //    int epochsWithoutImprovement = 0;
    //    int wastedEpochsToStopAt = 20;
    //    int minEpochs = 100;

    //    while (epochsWithoutImprovement >= wastedEpochsToStopAt)
    //    {
    //        (float loss, float accuracy) = TrainEpoch(currentEpoch, maxGradNorm, optimizer, criterion, tokenProvider, labels);

    //        if (accuracy < bestAccuracy)
    //        {
    //            // train for atleast X epochs before starting to count for bad epochs
    //            if (currentEpoch >= minEpochs)
    //                epochsWithoutImprovement++;
    //        }
    //        else
    //        {
    //            epochsWithoutImprovement = 0;
    //            bestState = state_dict();
    //            bestEpoch = currentEpoch;
    //        }

    //        currentEpoch++;
    //    }

    //    load_state_dict(bestState);
    //}

    //(float loss, float accuracy) TrainEpoch(int epoch, float maxGradNorm, AdamW optimizer, 
    //    BCEWithLogitsLoss criterion, IReplayTokenProvider tokenProvider, Tensor labels)
    //{
    //    float runningLoss = 0;
    //    optimizer.zero_grad();

    //    foreach (ReplayTokens tokens in tokenProvider)
    //    {
    //        using LstmData data = RunInference(tokens, true);
    //        using Tensor squeezedData = data.Data.squeeze(1);
    //        using Tensor loss = criterion.forward(squeezedData, labels);

    //        optimizer.zero_grad();
    //        loss.backward();

    //        _ = nn.utils.clip_grad_norm_(parameters(), maxGradNorm);

    //        using Tensor optimizerStep = optimizer.step();

    //        runningLoss += loss.item<float>();
    //    }

    //    runningLoss = 0;
    //    float accuracy = 0;

    //    foreach (ReplayTokens tokens in tokenProvider)
    //    {
    //        using LstmData data = RunInference(tokens, false);
    //        using Tensor squeezedData = data.Data.squeeze(1);
    //        using Tensor loss = criterion.forward(squeezedData, labels);

    //        runningLoss += loss.item<float>();
    //    }

    //    return (runningLoss, accuracy);
    //}
}
