using osuVendetta.Core.Replays.Data;
using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using System.Runtime.InteropServices;
using TorchSharp.Utils;
using System.Text;
using osuVendetta.Core.IO;
using static TorchSharp.torch.optim;
using osuVendetta.Core.Utility;

namespace osuVendetta.AntiCheatModel128x3;

public enum AntiCheatClass
{
    Normal = 0,
    Relax = 1
}

public class TrainReplayTokens : ReplayTokens
{
    public required AntiCheatClass Class { get; set; }
}

public class AntiCheatModel128x3 : Module<LstmData, LstmData>, IAntiCheatModel
{
    public DeviceType Device { get; private set; }
    public AntiCheatModelConfig Config { get; }

    readonly LSTM _lstm;
    readonly Dropout _dropOut;
    readonly Linear _fc;

    public AntiCheatModel128x3() : base("128x3 Anticheat Model")
    {
        Config = ConfigProvider.CreateConfig();

        _lstm = LSTM(
            inputSize: Config.InputSize,
            hiddenSize: Config.HiddenSize,
            numLayers: Config.LayerCount,
            batchFirst: true,
            bidirectional: true,
            dropout: Config.Dropout);

        _dropOut = Dropout(Config.Dropout);
        _fc = Linear(Config.HiddenSize * 2, Config.OutputSize);

        RegisterComponents();


#if WIN_CUDA_RELEASE || WIN_CUDA_DEBUG || LINUX_CUDA_DEBUG || LINUX_CUDA_RELEASE
        SetDevice(DeviceType.CUDA);
#endif
    }

    public AntiCheatModelResult RunInference(ReplayTokens tokens)
    {
        if (training)
            eval();

        using LstmData data = RunInference(tokens, false, null);

        return new AntiCheatModelResult
        {
            Segments = data.Data.ToArray<float>()
        };
    }

    // TODO: replace IEnumerable
    public interface IReplayTokenProvider : IEnumerable<TrainReplayTokens>
    {
        Dictionary<AntiCheatClass, int> GetReplayCount();
    }

    Tensor CreateTrainingLabels()
    {
        return tensor([
            (int)AntiCheatClass.Normal,
            (int)AntiCheatClass.Relax
        ]);
    }

    Tensor CreateClassWeights(Dictionary<AntiCheatClass, int> replayCount)
    {
        int[] classCounts = replayCount.OrderBy(kvp => (int)kvp.Key).Select(kvp => kvp.Value).ToArray();
        int totalClassCounts = classCounts.Sum();
        float[] classWeights = classCounts.Select(v => (float)(totalClassCounts / v)).ToArray();

        return tensor(classWeights);
    }


    public void RunTraining(IReplayTokenProvider tokenProvider, IProgressReporter progressReporter)
    {
        // implement prodigy https://github.com/konstmish/prodigy/blob/main/prodigyopt/prodigy.py
        // implement hyperparameter search
        // implement progress reports

        if (!training)
            train();

        Dictionary<AntiCheatClass, int> replayCount = tokenProvider.GetReplayCount();

        using Tensor labels = CreateTrainingLabels();
        using Tensor classWeights = CreateClassWeights(replayCount);
        using BCEWithLogitsLoss criterion = new BCEWithLogitsLoss(pos_weights: classWeights);
        using AdamW optimizer = new AdamW(parameters(), lr: 0.002, weight_decay: 1e-3);
        //using Prodigy optimizer = new Prodigy();

        optim.lr_scheduler.LRScheduler scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode: "min", factor: 0.5, patience: 5, verbose: true, min_lr: [1e-6]);

        TrainEpochs(optimizer, criterion, tokenProvider, labels);
    }

    void TrainEpochs(AdamW optimizer, BCEWithLogitsLoss criterion, 
        IReplayTokenProvider tokenProvider, Tensor labels)
    {
        float maxGradNorm = 1;

        Dictionary<string, Tensor> bestState = state_dict();
        float bestAccuracy = 0;
        int bestEpoch = 0;
        int currentEpoch = 0;
        int epochsWithoutImprovement = 0;
        int wastedEpochsToStopAt = 20;
        int minEpochs = 100;

        while (epochsWithoutImprovement >= wastedEpochsToStopAt)
        {
            (float loss, float accuracy) = TrainEpoch(currentEpoch, maxGradNorm, optimizer, criterion, tokenProvider, labels);

            if (accuracy < bestAccuracy)
            {
                // train for atleast X epochs before starting to count for bad epochs
                if (currentEpoch >= minEpochs)
                    epochsWithoutImprovement++;
            }
            else
            {
                epochsWithoutImprovement = 0;
                bestState = state_dict();
                bestEpoch = currentEpoch;
            }

            currentEpoch++;
        }

        load_state_dict(bestState);
    }

    (float loss, float accuracy) TrainEpoch(int epoch, float maxGradNorm, AdamW optimizer, 
        BCEWithLogitsLoss criterion, IReplayTokenProvider tokenProvider, Tensor labels)
    {
        float runningLoss = 0;
        optimizer.zero_grad();

        foreach (ReplayTokens tokens in tokenProvider)
        {
            using LstmData data = RunInference(tokens, true);
            using Tensor squeezedData = data.Data.squeeze(1);
            using Tensor loss = criterion.forward(squeezedData, labels);

            optimizer.zero_grad();
            loss.backward();

            _ = nn.utils.clip_grad_norm_(parameters(), maxGradNorm);

            using Tensor optimizerStep = optimizer.step();

            runningLoss += loss.item<float>();
        }

        runningLoss = 0;
        float accuracy = 0;

        foreach (ReplayTokens tokens in tokenProvider)
        {
            using LstmData data = RunInference(tokens, false);
            using Tensor squeezedData = data.Data.squeeze(1);
            using Tensor loss = criterion.forward(squeezedData, labels);

            runningLoss += loss.item<float>();
        }

        return (runningLoss, accuracy);
    }

    public void Load(Stream modelSafetensors)
    {
        Safetensors safetensors = Safetensors.Load(modelSafetensors);
        Dictionary<string, Tensor> stateDict = safetensors.ToStateDict();

        (IList<string> missingKeys, IList<string> unexpectedKeys) = load_state_dict(stateDict);

        foreach (string missingKey in missingKeys)
            Console.WriteLine($"Missing key: {missingKey}");

        foreach (string unexpectedKey in unexpectedKeys)
            Console.WriteLine($"Unexpected key: {unexpectedKey}");

        if (missingKeys.Count > 0 || unexpectedKeys.Count > 0)
        {
            Console.WriteLine("Unable to continue. Fix the model then retry.");
            Console.ReadLine();
            Environment.Exit(0);
        }

        _to(Device, 0, false);
    }

    public void SetDevice(DeviceType device)
    {
        Device = device;
        _to(Device, 0, false);
    }

    public override LstmData forward(LstmData input)
    {
        //if (input.device_type != Device)
        //input = input.to(Device);

        //using LstmData lstm = _lstm.forward(input);
        //using Tensor mean = lstm.Data.mean([1]);
        //using Tensor squeezed = mean.squeeze(1);
        //using Tensor dropOut = _dropOut.forward(squeezed);
        //using Tensor fc = _fc.forward(dropOut);

        using LstmData lstm = _lstm.forward(input.Data, input.HiddenState);
        using Tensor lstmData = lstm.Data.select(1, -1);
        using Tensor dropOut = _dropOut.forward(lstmData);
        using Tensor fc = _fc.forward(dropOut);

        Tensor output = sigmoid(fc);
        (Tensor H0, Tensor C0)? hiddenState = lstm.DetachHiddenState();

        return new LstmData(output, hiddenState);
    }

    LstmData RunInference(ReplayTokens tokens, bool isTraining, (Tensor H0, Tensor C0)? hiddenStates = null)
    {
        if (isTraining)
        {
            if (!training)
                train();
        }
        else
        {
            if (training)
                eval();
        }

        torch.Device device = torch.device(Device);

        int batchCount = (int)Math.Ceiling((float)tokens.Tokens.Length / Config.TotalFeatureSizePerChunk);
        using Tensor input = tensor(tokens.Tokens,
                                    dimensions: [batchCount, Config.StepsPerChunk, Config.FeaturesPerStep],
                                    device: device);

        return forward(new LstmData(input, hiddenStates));
    }
}
