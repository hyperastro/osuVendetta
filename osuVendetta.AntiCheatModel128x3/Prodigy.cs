using static TorchSharp.torch;
using TorchSharp.Modules;

namespace osuVendetta.AntiCheatModel128x3;

public class Prodigy : OptimizerHelper, optim.IBetas
{
    public (double, double) Betas { get; set; }

    public override Tensor step(Func<Tensor>? closure = null)
    {
        Tensor? loss = null;

        if (closure is not null)
            loss = closure();

        float dDenom = 0;
        ParamGroup paramGroups = _parameter_groups[0];

        return base.step(closure);
    }
}