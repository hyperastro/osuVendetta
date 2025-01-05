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

namespace osuVendetta.AntiCheatModel128x3;

public class AntiCheatModel128x3 : Module<Tensor, Tensor>, IAntiCheatModel
{
    class LstmResult : IDisposable
    {
        public bool IsDisposed { get; private set; }

        /// <summary>
        /// Item1 of <see cref="TorchSharp.Modules.LSTM.forward(Tensor, ValueTuple{Tensor, Tensor}?)"/>
        /// </summary>
        public Tensor IntPtr { get; init; }
        /// <summary>
        /// Item2 of <see cref="TorchSharp.Modules.LSTM.forward(Tensor, ValueTuple{Tensor, Tensor}?)"/>
        /// </summary>
        public Tensor Hn { get; init; }
        /// <summary>
        /// Item3 of <see cref="TorchSharp.Modules.LSTM.forward(Tensor, ValueTuple{Tensor, Tensor}?)"/>
        /// </summary>
        public Tensor Cn { get; init; }

        public LstmResult((Tensor intPtr, Tensor hn, Tensor cn) forwardResult)
        {
            IntPtr = forwardResult.intPtr;
            Hn = forwardResult.hn;
            Cn = forwardResult.cn;
        }

        public static implicit operator LstmResult((Tensor intPtr, Tensor hn, Tensor cn) forwardResult)
        {
            return new LstmResult(forwardResult);
        }

        public void Dispose()
        {
            if (IsDisposed)
                return;

            IntPtr.Dispose();
            Hn.Dispose();
            Cn.Dispose();
            IsDisposed = true;
        }

    }

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

        torch.Device device = torch.device(Device);

        int batchSize = (int)Math.Ceiling((float)tokens.Tokens.Length / Config.TotalFeatureSizePerChunk);
        using Tensor input = tensor(tokens.Tokens, 
                                    dimensions: [batchSize, Config.StepsPerChunk, Config.FeaturesPerStep], 
                                    device: device);

        using Tensor output = forward(input);
        using Tensor flattened = output.flatten();
        using TensorAccessor<float> flattenedAccessor = flattened.data<float>();

        return new AntiCheatModelResult
        {
            Segments = flattenedAccessor.ToArray()
        };
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

        _to(Device, 0, false);

        Console.ReadLine();
    }

    public void SetDevice(DeviceType device)
    {
        Device = device;
        _to(Device, 0, false);
    }

    public override Tensor forward(Tensor input)
    {
        if (input.device_type != Device)
            input = input.to(Device);

        using LstmResult lstmTensor = _lstm.forward(input);
        using Tensor lstmMean = lstmTensor.IntPtr.mean([1]);
        using Tensor lstmSqueezed = lstmMean.squeeze(1);
        using Tensor dropOutTensor = _dropOut.forward(lstmSqueezed);
        using Tensor logits = _fc.forward(dropOutTensor);

        return sigmoid(logits);
    }
}
