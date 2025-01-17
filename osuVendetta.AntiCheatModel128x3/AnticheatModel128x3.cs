﻿using osuVendetta.Core.Replays.Data;
using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using System.Runtime.InteropServices;
using TorchSharp.Utils;
using System.Text;
using osuVendetta.Core.IO.Safetensors;

namespace osuVendetta.AntiCheatModel128x3;

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


    public LstmData RunInference(ReplayTokens tokens, bool isTraining, (Tensor H0, Tensor C0)? hiddenStates = null)
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

        int batchCount = (int)Math.Ceiling((float)tokens.Tokens.Length / Config.TotalFeatureSizePerChunk);
        using Tensor input = tensor(tokens.Tokens,
                                    dimensions: [batchCount, Config.StepsPerChunk, Config.FeaturesPerStep]);

        using LstmData lstmInput = new LstmData(input, hiddenStates);

        return forward(lstmInput);
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
    }

    public void SetDevice(DeviceType device)
    {
        set_default_device(torch.device(device));
        Device = device;
        Module m = _to(Device, 0, false);
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
        using Tensor lstmData = lstm.Data[TensorIndex.Colon, -1, TensorIndex.Colon];
        using Tensor dropOut = _dropOut.forward(lstmData);

        //using Tensor fc = _fc.forward(dropOut);
        //Tensor output = sigmoid(fc);

        Tensor output = _fc.forward(dropOut);
        (Tensor H0, Tensor C0)? hiddenState = lstm.DetachHiddenState();

        return new LstmData(output, hiddenState);
    }
}
