using osuVendetta.Core.Replays.Data;
using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using System.Runtime.InteropServices;
using TorchSharp.Utils;

namespace osuVendetta.AntiCheatModel128x3;

public class AnticheatModel128x3 : Module<Tensor, Tensor>, IAntiCheatModel
{
    public DeviceType Device { get; private set; }
    public AntiCheatModelConfig Config { get; }

    readonly LSTM lstm;
    readonly Dropout dropOut;
    readonly Linear fc;

    public AnticheatModel128x3() : base("128x3 Anticheat Model")
    {
        Config = ConfigProvider.CreateConfig();

        lstm = LSTM(
            inputSize: Config.InputSize,
            hiddenSize: Config.HiddenSize,
            numLayers: Config.LayerCount,
            batchFirst: true,
            bidirectional: true,
            dropout: Config.Dropout);

        dropOut = Dropout(Config.Dropout);
        fc = Linear(Config.HiddenSize * 2, Config.OutputSize);

        RegisterComponents();

#if WIN_CUDA_RELEASE || WIN_CUDA_DEBUG || LINUX_CUDA_DEBUG || LINUX_CUDA_RELEASE
        Device = DeviceType.CUDA;
        _to(Device, 0, false);
#endif
    }

    public AntiCheatModelResult RunInference(ReplayTokens tokens)
    {
        if (training)
            eval();

        torch.Device device = torch.device(Device);

        using Tensor input = tensor(tokens.Tokens, device: device);
        using Tensor output = forward(input);
        using Tensor flattened = output.flatten();
        using TensorAccessor<float> flattenedAccessor = flattened.data<float>();

        return new AntiCheatModelResult
        {
            ReplayName = tokens.ReplayName,
            Segments = flattenedAccessor.ToArray()
        };
    }

    public void Load(Stream modelWeights)
    {
        load(modelWeights);
    }

    public IAntiCheatModel ToDevice(DeviceType device)
    {
        AnticheatModel128x3 model = this.to(device);
        model.Device = device;

        return model;
    }

    public override Tensor forward(Tensor input)
    {
        if (input.device_type != Device)
            input = input.to(Device);

        using LstmResult lstmTensor = lstm.forward(input);
        using Tensor lstmMean = lstmTensor.IntPtr.mean([1]);
        using Tensor lstmSqueezed = lstmMean.squeeze(1);
        using Tensor dropOutTensor = dropOut.forward(lstmSqueezed);
        using Tensor logits = fc.forward(dropOutTensor);

        return sigmoid(logits);
    }

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
}

//public static class TensorExtension
//{
//    public static Tensor ChainAndDispose(this Tensor input, Func<Tensor, Tensor> output)
//    {
//        try
//        {
//            return output(input);
//        }
//        finally
//        {
//            input.Dispose();
//        }
//    }
//}
