using static TorchSharp.torch;

namespace osuVendetta.AntiCheatModel128x3;

public class LstmData : IDisposable
{
    public bool IsDisposed { get; private set; }

    /// <summary>
    /// Output of <see cref="TorchSharp.Modules.LSTM.forward(Tensor, ValueTuple{Tensor, Tensor}?)"/>
    /// </summary>
    public Tensor Data { get; init; }
    /// <summary>
    /// Hidden state of the LSTM
    /// </summary>
    public (Tensor H0, Tensor C0)? HiddenState { get; init; }

    public LstmData(Tensor data, (Tensor H0, Tensor C0)? hiddenState)
    {
        Data = data;
        HiddenState = hiddenState;
    }

    public static implicit operator LstmData((Tensor data, Tensor hn, Tensor cn) lstmResult)
    {
        return new LstmData(lstmResult.data, (lstmResult.hn, lstmResult.cn));
    }

    public (Tensor H0, Tensor C0)? DetachHiddenState()
    {
        if (HiddenState is null)
            return null;

        return (HiddenState.Value.H0.detach(), HiddenState.Value.C0.detach());
    }

    public void Dispose()
    {
        if (IsDisposed)
            return;

        Data.Dispose();
        
        if (HiddenState is not null)
        {
            HiddenState.Value.H0.Dispose();
            HiddenState.Value.C0.Dispose();
        }

        IsDisposed = true;
    }

}