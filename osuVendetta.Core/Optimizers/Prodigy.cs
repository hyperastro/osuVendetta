using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp.Utils;
using System.Text.Encodings.Web;
using TorchSharp;
using Tensorboard;
using static TorchSharp.torch.distributions.constraints;
using OptimizerOptions = TorchSharp.Modules.OptimizerOptions;

namespace osuVendetta.Core.Optimizers;

public class Prodigy : OptimizerHelper, optim.IBetas
{
    public (double, double) Betas
    {
        get
        {
            Options defaults = (Options)_defaults;
            return (defaults.Beta1, defaults.Beta2);
        }
        set
        {
            Options defaults = (Options)_defaults;
            defaults.Beta1 = (float)value.Item1;
            defaults.Beta2 = (float)value.Item2;
        }
    }

    public Prodigy(IEnumerable<Parameter> parameters, float learningRate = 1, float beta1 = .9f, float beta2 = .999f, float? beta3 = null,
        float eps = 1e-8f, float weightDecay = 0, bool decouple = true, bool useBiasCorrection = false,
        bool safeguardWarmup = false, float d0 = 1e-6f, float dCoef = 1f, float growthRate = float.PositiveInfinity,
        long sliceP = 1) :
        this([ new ParamGroup(parameters) ], learningRate, beta1, beta2, beta3, eps, weightDecay, decouple, useBiasCorrection, safeguardWarmup, 
            d0, dCoef, growthRate, sliceP)
    {
    }
    public Prodigy(IEnumerable<ParamGroup> parameters, float learningRate = 1, float beta1 = .9f, float beta2 = .999f, float? beta3 = null,
        float eps = 1e-8f, float weightDecay = 0, bool decouple = true, bool useBiasCorrection = false,
        bool safeguardWarmup = false, float d0 = 1e-6f, float dCoef = 1f, float growthRate = float.PositiveInfinity,
        long sliceP = 1)
    {
        _defaults = new Options
        {
            LearningRate = learningRate,
            Beta1 = beta1,
            Beta2 = beta2,
            Beta3 = beta3,
            Eps = eps,
            WeightDecay = weightDecay,
            D = d0,
            D0 = d0,
            DMax = d0,
            DNumerator = 0,
            DCoef = dCoef,
            K = 0,
            GrowthRate = growthRate,
            UseBiasCorrection = useBiasCorrection,
            Decouple = decouple,
            SafeguardWarmup = safeguardWarmup,
            // FSDP is not supported
            FsdpInUse = false,
            SliceP = sliceP
        };

        _parameter_groups = new List<TorchSharp.Modules.ParamGroup>();
        foreach (ParamGroup group in parameters)
            add_param_group(group);
    }

    public override Tensor step(Func<Tensor>? closure = null)
    {
        ParamGroup group = (ParamGroup)_parameter_groups[0];

        if (group.Options.FsdpInUse)
            throw new InvalidOperationException("Sharded/Distributed parameters currently not implemented");

        Tensor? loss = null;

        if (closure is not null)
            loss = closure();

        float dDenom = 0;

        bool useBiasCorrection = group.Options.UseBiasCorrection;
        Tensor beta1 = group.Options.Beta1;
        Tensor beta2 = group.Options.Beta2;

        Tensor beta3;
        if (group.Options.Beta3.HasValue)
            beta3 = group.Options.Beta3.Value;
        else
            beta3 = beta2.sqrt();

        Tensor k = group.Options.K;

        Tensor d = group.Options.D;
        Tensor dMax = group.Options.DMax;
        Tensor dCoef = group.Options.DCoef;
        float lr = (float)_parameter_groups.Max(p => ((Options)p.Options).LearningRate ?? 1);

        Tensor biasCorrection;
        if (useBiasCorrection)
            biasCorrection = (1 - beta2.pow(k + 1)) / (1 - beta1.pow(k + 1));
        else
            biasCorrection = 1f;

        Tensor dlr = d * lr * biasCorrection;

        float growthRate = group.Options.GrowthRate;
        bool decouple = group.Options.Decouple;
        //bool fsdpInUse = group.Options.FsdpInUse;

        Tensor dNumerator = group.Options.DNumerator * beta3;
        Tensor deltaNumerator = 0;

        return _step<ParamGroup>(group =>
        {
            float decay = group.Options.WeightDecay;
            k = group.Options.K;
            float eps = group.Options.Eps;
            float groupLr = (float)(group.Options.LearningRate ?? 1);
            float d0 = group.Options.D0;
            bool safeguardWarmup = group.Options.SafeguardWarmup;
            long sliceP = group.Options.SliceP;

            if (groupLr != 0 && groupLr != lr)
                throw new InvalidOperationException("Setting different lr values in different parameter groups is only supported for values of 0");

            foreach (Parameter p in group.Parameters)
            {
                if (p.grad is null)
                    continue;

                // Skyfly: I don't think this is needed, if we use fsdp we will set it to true anyway
                // Skyfly: atleast i have no idea how to implement this right now
                //if (hasattr(p, "_fsdp_flattened"))
                //    fsdp_in_use = True;

                Tensor grad = p.grad;

                // Apply weight decay (coupled variant)
                if (decay != 0 && !decouple)
                    grad.add_(grad, decay);

                // State initialization
                State state = (State)_state[p.Handle];

                if (!state.Step.HasValue)
                {
                    state.Step = 0;

                    state.S = state.S.DisposeAndReplace(zeros_like(p.flatten().slice(0, 0, -1, sliceP)).detach(), true);

                    Tensor pAny = p.any();
                    bool[] pAnyData = pAny.ToArray<bool>();

                    if (pAnyData.Any())
                        state.P0 = state.P0.DisposeAndReplace(p.flatten().slice(0, 0, -1, sliceP).detach().clone(), true);
                    else
                        // All values are zero, so save VRAM with a zero-tensor
                        state.P0 = state.P0.DisposeAndReplace(tensor(0, device: p.device, dtype: p.dtype), true);

                    // Exponential moving average of gradient values
                    if (beta1.item<float>() > 0)
                        state.ExpAvg = state.ExpAvg.DisposeAndReplace(zeros_like(p).detach(), true);

                    // Exponential moving average of squared gradient values
                    state.ExpAvgSq = state.ExpAvgSq.DisposeAndReplace(zeros_like(p).detach(), true);
                }

                // Skyfly: ExpAvgSq and P0 get their value set in the if scope at line 136~ if (!state.Step.HasValue)
                Tensor expAvgSq = state.ExpAvgSq!;
                Tensor s = state.Step.Value;
                Tensor p0 = state.P0!;
                Tensor expAvg;

                if (groupLr > 0)
                {
                    // we use d / d0 instead of just d to avoid getting values that are too small

                    Tensor slicedGrad = grad.flatten().slice(0, 0, -1, sliceP);
                    deltaNumerator += (d / d0) * dlr * dot(slicedGrad, p0 - p0.flatten().slice(0, 0, -1, sliceP));

                    //# Adam EMA updates
                    if (beta1.item<float>() > 0)
                    {
                        var x = d * (1 - beta1);
                        expAvg = state.ExpAvg!;
                        expAvg!.mul_(beta1).add_(grad, (d * (1 - beta1)).ToScalar());
                    }

                    expAvgSq.mul_(beta2).addcmul_(grad, grad, (d * d * (1 - beta2)).ToScalar());

                    if (safeguardWarmup)
                        s.mul_(beta3).add_(slicedGrad, (d / d0 * d).ToScalar());
                    else
                        s.mul_(beta3).add_(slicedGrad, (d / d0 * dlr).ToScalar());

                    dDenom += s.abs().sum().item<float>();
                }
            }

            Tensor dHat = d;
            //Tensor distTensor;
            Tensor? globalDNumerator = null;
            Tensor? globalDDenom = null;

            if (lr > 0)
            {
                // Unless we have an fsdp implementation, we will never run this
                //if (fsdpInUse)
                //{
                //    distTensor = tensor([deltaNumerator, dDenom], device: device(DeviceType.CUDA));
                //    dist.all_reduce(dist_tensor, op = dist.ReduceOp.SUM);
                //    globalDNumerator = dNumerator + distTensor[0];
                //    globalDDenom = distTensor[1];
                //}
                //else
                //{
                //    globalDNumerator = dNumerator + deltaNumerator;
                //    globalDDenom = dDenom;
                //}

                globalDNumerator = dNumerator + deltaNumerator;
                globalDDenom = dDenom;
                dHat = dCoef * globalDDenom / globalDDenom;

                // Skyfly: standard implementation is no null check, just equal
                // Skyfly: the thing is, D could be null here, so let's make sure to check it
                if (group.D is not null && d.equal(group.D).item<bool>())
                {
                    d = max(d, dHat);
                }

                dMax = max(dMax, dHat);
                d = min(dMax, d * growthRate);
            }

            foreach (ParamGroup pgroup in _parameter_groups)
            {
                pgroup.DNumerator = pgroup.DNumerator.DisposeAndReplace(globalDNumerator, true);
                pgroup.DDenom = pgroup.DDenom.DisposeAndReplace(globalDNumerator, true);
                pgroup.D = pgroup.D.DisposeAndReplace(globalDNumerator, true);
                pgroup.DMax = pgroup.DMax.DisposeAndReplace(globalDNumerator, true);
                pgroup.DHat = pgroup.DHat.DisposeAndReplace(globalDNumerator, true);

                foreach (Parameter p in pgroup.Parameters)
                {
                    if (p.grad is null)
                        continue;

                    Tensor grad = p.grad;

                    State state = (State)_state[p.Handle];

                    // Skyfly: ExpAvgSq and ExpAvg get their value set in the if scope at line 136~ if (!state.Step.HasValue)

                    Tensor expAvgSq = state.ExpAvgSq!;

                    state.Step++;

                    Tensor denom = expAvgSq.sqrt().add(d * eps);

                    // Apply weight decay (decoupled variant)
                    if (decay != 0 && decouple)
                        p.add_(p, (decay * dlr).ToScalar());

                    // Take step
                    if (beta1.item<float>() > 0)
                    {
                        Tensor expAvg = state.ExpAvg!;
                        p.addcdiv_(expAvg, denom, dlr.ToScalar());
                    }
                    else
                    {
                        p.addcdiv_(grad, denom, (dlr * d).ToScalar());
                    }
                }

                pgroup.K = k.item<float>() + 1;
            }
        }, closure);
    }


    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        foreach ((nint, OptimizerState) statePair in _state)
            ((State)statePair.Item2).Dispose();
    }

    public override void add_param_group(TorchSharp.Modules.ParamGroup paramGroup)
    {
        Options defaultOptions = (Options)_defaults;
        Options? options = (Options?)paramGroup.Options;

        if (options is null)
            paramGroup.Options = options = defaultOptions.CreateCopy();

        paramGroup.Options.InitialLearningRate = defaultOptions.LearningRate ?? 1;


        _parameter_groups.Add(paramGroup);

        foreach (Parameter parameter in paramGroup.Parameters)
        {
            State state = new State(parameter);
            _state[parameter.Handle] = state;
            state.Initialize(options);
        }
    }


    public class State : OptimizerState, IDisposable
    {
        public bool IsDisposed { get; private set; }

        public long? Step { get; set; }
        public Tensor? S { get; set; }
        public Tensor? P0 { get; set; }
        public Tensor? ExpAvg { get; set; }
        public Tensor? ExpAvgSq { get; set; }

        public State(Parameter parameter) : base(parameter)
        {
        }

        public void Dispose()
        {
            if (IsDisposed)
                return;

            Step = null;

            S?.Dispose();
            S = null;

            P0?.Dispose();
            P0 = null;

            ExpAvg?.Dispose();
            ExpAvg = null;

            ExpAvgSq?.Dispose();
            ExpAvgSq = null;

            IsDisposed = true;
            GC.SuppressFinalize(this);
        }

        public override void to(Device device)
        {
            S?.to(device);
            P0?.to(device);
            ExpAvg?.to(device);
            ExpAvgSq?.to(device);
        }

        public override void Initialize(OptimizerOptions options)
        {
            // Skyfly: State is initialized in prodigy itself
        }

        public override void LoadStateDict(BinaryReader reader)
        {
            Step = reader.ReadInt64();

            S = zeros_like(_parameter).DetachFromDisposeScope();
            S.Load(reader);

            P0 = zeros_like(_parameter).DetachFromDisposeScope();
            P0.Load(reader);

            ExpAvg = zeros_like(_parameter).DetachFromDisposeScope();
            ExpAvg.Load(reader);

            ExpAvgSq = zeros_like(_parameter).DetachFromDisposeScope();
            ExpAvgSq.Load(reader);
        }

        public override void LoadStateDict(OptimizerState source)
        {
            State srcState = (State)source;

            Step = srcState.Step;
            S = srcState.S;
            P0 = srcState.P0;
            ExpAvg = srcState.ExpAvg;
            ExpAvgSq = srcState.ExpAvgSq;
        }

        public override void SaveStateDict(BinaryWriter writer)
        {
            // Skyfly: for now assume everything has a value

            writer.Write(Step!.Value);
            S!.Save(writer);
            P0!.Save(writer);
            ExpAvg!.Save(writer);
            ExpAvgSq!.Save(writer);
        }
    }

    public class Options : OptimizerOptions
    {
        public required float Beta1 { get; set; }
        public required float Beta2 { get; set; }
        public required float? Beta3 { get; set; }
        public required float Eps { get; set; }
        public required float WeightDecay { get; set; }
        public required bool Decouple { get; set; }
        public required bool UseBiasCorrection { get; set; }
        public required bool SafeguardWarmup { get; set; }
        public required float D0 { get; set; }
        public required float DMax { get; set; }
        public required float D { get; set; }
        public required float DCoef { get; set; }
        public required float DNumerator { get; set; }
        public required float GrowthRate { get; set; }
        public required bool FsdpInUse { get; set; }
        public required long SliceP { get; set; }
        public required float K { get; set; }

        public Options CreateCopy()
        {
            return new Options
            {
                Beta1 = Beta1,
                Beta2 = Beta2,
                Beta3 = Beta3,
                Eps = Eps,
                WeightDecay = WeightDecay,
                Decouple = Decouple,
                UseBiasCorrection = UseBiasCorrection,
                SafeguardWarmup = SafeguardWarmup,
                D0 = D0,
                DMax = DMax,
                D = D,
                DCoef = DCoef,
                DNumerator = DNumerator,
                GrowthRate = GrowthRate,
                FsdpInUse = FsdpInUse,
                SliceP = SliceP,
                K = K
            };
        }
    }

    public class ParamGroup : ParamGroup<Options>, optim.IBetas/*, IDisposable*/
    {
        public bool IsDisposed { get; private set; }

        public (double, double) Betas
        {
            get => (Options.Beta1, Options.Beta2);
            set
            {
                Options.Beta1 = (float)value.Item1;
                Options.Beta2 = (float)value.Item2;
            }
        }
        public Tensor? DNumerator { get; set; }
        public Tensor? DDenom { get; set; }
        public Tensor? D { get; set; }
        public Tensor? DMax { get; set; }
        public Tensor? DHat { get; set; }
        public float WeightDecay { get; set; }
        public float K { get; set; }
        public float EPS { get; set; }

        //public ParamGroup(IEnumerable<Parameter> parameters, Options options) : base(parameters, options) 
        //{ 

        //}

        public ParamGroup(IEnumerable<Parameter> parameters, float learningRate = 1, float beta1 = .9f,
            float beta2 = .999f, float? beta3 = null, float eps = 1e-8f, float weightDecay = 0f,
            bool decouple = true, bool useBiasCorrection = false, bool safeguardWarmup = false, float d0 = 1e-6f,
            float dCoef = 1f, float growthRate = float.PositiveInfinity, bool fsdpInUse = false, long sliceP = 1) :
            base(parameters, new Options
            {
                LearningRate = 1,
                Beta1 = beta1,
                Beta2 = beta2,
                Beta3 = beta3,
                Eps = eps,
                WeightDecay = weightDecay,
                Decouple = decouple,
                UseBiasCorrection = useBiasCorrection,
                SafeguardWarmup = safeguardWarmup,
                D = d0,
                D0 = d0,
                DMax = d0,
                DNumerator = 0,
                DCoef = dCoef,
                K = 0,
                GrowthRate = growthRate,
                FsdpInUse = fsdpInUse,
                SliceP = sliceP
            })
        {

        }

        //public void Dispose()
        //{

        //    GC.SuppressFinalize(this);
        //}
    }
}