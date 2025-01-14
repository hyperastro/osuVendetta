using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp.Utils;
using System.Text.Encodings.Web;
using TorchSharp;
using Tensorboard;
using static TorchSharp.torch.distributions.constraints;
using OptimizerOptions = TorchSharp.Modules.OptimizerOptions;
using OsuParsers.Enums.Database;
using static Tensorboard.CostGraphDef.Types;
using static TorchSharp.torch.optim;
using System;
using System.ComponentModel.Design;

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

    //		global_d_numerator = d_numerator + delta_numerator
    //		global_d_denom = d_denom
    float _globalDNumerator;
    float _globalDDenom;
    float _d0;

    public override Tensor step(Func<Tensor>? closure = null)
    {
        //loss = None
        //if closure is not None:
        //    loss = closure()
        Tensor? loss = null;

        if (closure is not null)
            loss = closure();

        //d_denom = 0.0
        float dDenom = 0f;

        //group = self.param_groups[0]
        ParamGroup group0 = (ParamGroup)_parameter_groups[0];
        //use_bias_correction = group['use_bias_correction']
        bool useBiasCorrection = group0.UseBiasCorrection;
        //beta1, beta2 = group['betas']
        (float beta1, float beta2) = group0.BetasFloat;
        //beta3 = group['beta3']
        //if beta3 is None:
        //	beta3 = math.sqrt(beta2)
        float beta3 = group0.Beta3 ?? (float)Math.Sqrt(beta2);
        //k = group['k']
        float k = group0.K;

        //d = group['d']
        float d = group0.D;
        //d_max = group['d_max']
        float dMax = group0.DMax;
        //d_coef = group['d_coef']
        float dCoef = group0.DCoef;
        //lr = max(group['lr'] for group in self.param_groups)
        float lr = (float)_parameter_groups.Max(g => g.LearningRate);
        //if use_bias_correction:
        //	bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        //else:
        //	bias_correction = 1
        float biasCorrection;
        if (useBiasCorrection)
            biasCorrection = (float)(Math.Pow(Math.Pow(beta2, k + 1), .5f) / (1 - Math.Pow(beta1, k + 1)));
        else
            biasCorrection = 1f;

        //dlr = d*lr*bias_correction
        float dlr = d * lr * biasCorrection;

        //growth_rate = group['growth_rate']
        float growthRate = group0.GrowthRate;
        //decouple = group['decouple']
        bool decouple = group0.Decouple;
        //fsdp_in_use = group['fsdp_in_use']

        //d_numerator = group['d_numerator']
        //d_numerator *= beta3
        float dNumerator = group0.DNumerator * beta3;
        //delta_numerator = 0.0
        float deltaNumerator = 0f;

        //for group in self.param_groups:
        foreach (ParamGroup group in _parameter_groups)
        {
            //	decay = group['weight_decay']
            float decay = group.WeightDecay;
            //	k = group['k']
            k = group.K;
            //	eps = group['eps']
            float eps = group.EPS;
            //	group_lr = group['lr']
            float groupLr = (float)group.LearningRate;
            //	d0 = group['d0']
            _d0 = group.D0;
            //	safeguard_warmup = group['safeguard_warmup']
            bool safeguardWarmup = group.SafeguardWarmup;
            //	slice_p = group['slice_p']
            long sliceP = group.SliceP;

            //	if group_lr not in [lr, 0.0]:
            //		raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")
            if (groupLr != lr && groupLr != 0)
                throw new InvalidOperationException("Setting different lr values in different parameter groups is only supported for values of 0");

            //	for p in group['params']:
            foreach (Parameter p in group.Parameters)
            {
                //		if p.grad is None:
                //			continue
                if (p.grad is null)
                    continue;

                //		if hasattr(p, "_fsdp_flattened"):
                //			fsdp_in_use = True

                //		grad = p.grad.data
                Tensor grad = p.grad;

                //		# Apply weight decay (coupled variant)
                //		if decay != 0 and not decouple:
                //			grad.add_(p.data, alpha=decay)
                if (decay != 0 && !decouple)
                    grad.add_(p, alpha: decay);

                //		state = self.state[p]
                State state = (State)_state[p.Handle];

                //		# State initialization
                //		if 'step' not in state:
                if (state.Step is null)
                {
                    //			state['step'] = 0
                    state.Step = 0;

                    //			state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()
                    state.S?.Dispose();
                    state.S = zeros_like(p.flatten().slice(0, 0, -1, sliceP)).detach().MoveToOuterDisposeScope();
                    //			if p.any():
                    //				state['p0'] = p.flatten()[::slice_p].detach().clone()
                    //			else:
                    //				# All values are zero, so save VRAM with a zero-tensor
                    //				state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)
                    state.P0?.Dispose();
                    if (p.ToArray<bool>().Any())
                        state.P0 = p.flatten().slice(0, 0, -1, sliceP).detach().clone().MoveToOuterDisposeScope();
                    else
                        state.P0 = tensor(0, device: p.device, dtype: p.dtype).MoveToOuterDisposeScope();

                    //			# Exponential moving average of gradient values
                    //			if beta1 > 0:
                    //				state['exp_avg'] = torch.zeros_like(p.data).detach()
                    //			# Exponential moving average of squared gradient values
                    //			state['exp_avg_sq'] = torch.zeros_like(p.data).detach()
                    if (beta1 > 0)
                    {
                        state.ExpAvg?.Dispose();
                        state.ExpAvg = zeros_like(p).detach().MoveToOuterDisposeScope();
                    }

                    state.ExpAvgSq?.Dispose();
                    state.ExpAvgSq = zeros_like(p).detach().MoveToOuterDisposeScope();
                }

                //		exp_avg_sq = state['exp_avg_sq']
                Tensor expAvgSq = state.ExpAvgSq!;
                //		s = state['s']
                Tensor s = state.S!;
                //		p0 = state['p0']
                Tensor p0 = state.P0!;

                //		if group_lr > 0.0:
                if (groupLr > 0)
                {
                    //			# we use d / d0 instead of just d to avoid getting values that are too small
                    //			sliced_grad = grad.flatten()[::slice_p]
                    //			delta_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()
                    Tensor slicedGrad = grad.flatten().slice(0, 0, -1, sliceP);
                    deltaNumerator += (d / _d0) * dlr * dot(slicedGrad, p0 - p.flatten().slice(0, 0, -1, sliceP)).item<float>();

                    //			# Adam EMA updates
                    //			if beta1 > 0:
                    //				exp_avg = state['exp_avg']
                    //				exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    if (beta1 > 0)
                    {
                        Tensor expAvg = state.ExpAvg!;
                        expAvg.mul_(beta1).add_(grad, alpha: d * (1 - beta1));
                    }
                    //			exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))
                    expAvgSq.mul_(beta2).addcmul_(grad, grad, value: d * d * (1 - beta2));

                    //			if safeguard_warmup:
                    //				s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    //			else:
                    //				s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    if (safeguardWarmup)
                        s.mul_(beta3).add_(slicedGrad, alpha: (d / _d0) * d);
                    else
                        s.mul_(beta3).add_(slicedGrad, alpha: (d / _d0) * dlr);
                    //			d_denom += s.abs().sum().item()
                    dDenom += s.abs().sum().item<float>();

                    //	######
                }
            }
        }
        //d_hat = d
        float dHat = d;

        //# if we have not done any progres, return
        //# if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        //if d_denom == 0:
        //	return loss
        if (dDenom == 0)
            return loss;

        //if lr > 0.0:
        if (lr > 0)
        {
            //	if fsdp_in_use:
            //		dist_tensor = torch.zeros(2).cuda()
            //		dist_tensor[0] = delta_numerator
            //		dist_tensor[1] = d_denom
            //		dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            //		global_d_numerator = d_numerator + dist_tensor[0]
            //		global_d_denom = dist_tensor[1]
            //	else:
            //		global_d_numerator = d_numerator + delta_numerator
            //		global_d_denom = d_denom

            _globalDNumerator = dNumerator + deltaNumerator;
            _globalDDenom = dDenom;
        }

        //	d_hat = d_coef * global_d_numerator / global_d_denom
        dHat = dCoef * _globalDNumerator / _globalDDenom;
        //	if d == group['d0']:
        //		d = max(d, d_hat)
        if (d == group0.D)
            d = max(d, dHat).item<float>();
        //	d_max = max(d_max, d_hat)
        dMax = max(dMax, dHat).item<float>();
        //	d = min(d_max, d * growth_rate)
        d = min(dMax, d * growthRate).item<float>();

        //for group in self.param_groups:
        foreach (ParamGroup group in _parameter_groups)
        {
            //	group['d_numerator'] = global_d_numerator
            group.DNumerator = _globalDNumerator;
            //	group['d_denom'] = global_d_denom
            group.DDenom = _globalDDenom;
            //	group['d'] = d
            group.D = d;
            //	group['d_max'] = d_max
            group.DMax = dMax;
            //	group['d_hat'] = d_hat
            group.DHat = dHat;

            //	decay = group['weight_decay']
            float decay = group.WeightDecay;
            //	k = group['k']
            k = group.K;
            //	eps = group['eps']
            float eps = group.EPS;

            //	for p in group['params']:
            foreach (Parameter p in group.Parameters)
            {
                //		if p.grad is None:
                //			continue
                if (p.grad is null)
                    continue;

                //		grad = p.grad.data
                Tensor grad = p.grad;

                //		state = self.state[p]
                State state = (State)_state[p.Handle];
                //		exp_avg_sq = state['exp_avg_sq']
                Tensor expAvgSq = state.ExpAvgSq!;

                //		state['step'] += 1
                state.Step += 1;

                //		denom = exp_avg_sq.sqrt().add_(d * eps)
                Tensor denom = expAvgSq.sqrt().add_(d * eps);

                //		# Apply weight decay (decoupled variant)
                //		if decay != 0 and decouple:
                //			p.data.add_(p.data, alpha=-decay * dlr)
                if (decay != 0 && decouple)
                    p.add_(p, alpha: -decay * dlr);

                //		### Take step
                //		if beta1 > 0:
                //			exp_avg = state['exp_avg']
                //			p.data.addcdiv_(exp_avg, denom, value=-dlr)
                //		else:
                //			p.data.addcdiv_(grad, denom, value=-dlr * d)
                if (beta1 > 0)
                {
                    Tensor expAvg = state.ExpAvg!;
                    p.addcdiv_(expAvg, denom, value: -dlr);
                }
                else
                {
                    p.addcdiv_(grad, denom, value: -dlr * d);
                }
            }

            //	group['k'] = k + 1
            group.K = k + 1;
        }

        return loss;
    }

    //public override Tensor step(Func<Tensor>? closure = null)
    //{
    //    ParamGroup group = (ParamGroup)_parameter_groups[0];

    //    if (group.Options.FsdpInUse)
    //        throw new InvalidOperationException("Sharded/Distributed parameters currently not implemented");

    //    Tensor? loss = null;

    //    if (closure is not null)
    //        loss = closure();

    //    float dDenom = 0;

    //    bool useBiasCorrection = group.Options.UseBiasCorrection;
    //    Tensor beta1 = group.Options.Beta1;
    //    Tensor beta2 = group.Options.Beta2;

    //    Tensor beta3;
    //    if (group.Options.Beta3.HasValue)
    //        beta3 = group.Options.Beta3.Value;
    //    else
    //        beta3 = beta2.sqrt();

    //    Tensor k = group.Options.K;

    //    Tensor d = group.Options.D;
    //    Tensor dMax = group.Options.DMax;
    //    Tensor dCoef = group.Options.DCoef;
    //    float lr = (float)_parameter_groups.Max(p => ((Options)p.Options).LearningRate ?? 1);

    //    Tensor biasCorrection;
    //    if (useBiasCorrection)
    //        biasCorrection = (1 - beta2.pow(k + 1)) / (1 - beta1.pow(k + 1));
    //    else
    //        biasCorrection = 1f;

    //    Tensor dlr = d * lr * biasCorrection;

    //    float growthRate = group.Options.GrowthRate;
    //    bool decouple = group.Options.Decouple;
    //    //bool fsdpInUse = group.Options.FsdpInUse;

    //    Tensor dNumerator = group.Options.DNumerator * beta3;
    //    Tensor deltaNumerator = 0;

    //    return _step<ParamGroup>(group =>
    //    {
    //        float decay = group.Options.WeightDecay;
    //        k = group.Options.K;
    //        float eps = group.Options.Eps;
    //        float groupLr = (float)(group.Options.LearningRate ?? 1);
    //        float d0 = group.Options.D0;
    //        bool safeguardWarmup = group.Options.SafeguardWarmup;
    //        long sliceP = group.Options.SliceP;

    //        if (groupLr != 0 && groupLr != lr)
    //            throw new InvalidOperationException("Setting different lr values in different parameter groups is only supported for values of 0");

    //        foreach (Parameter p in group.Parameters)
    //        {
    //            if (p.grad is null)
    //                continue;

    //            // Skyfly: I don't think this is needed, if we use fsdp we will set it to true anyway
    //            // Skyfly: atleast i have no idea how to implement this right now
    //            //if (hasattr(p, "_fsdp_flattened"))
    //            //    fsdp_in_use = True;

    //            Tensor grad = p.grad;

    //            // Apply weight decay (coupled variant)
    //            if (decay != 0 && !decouple)
    //                grad.add_(grad, decay);

    //            // State initialization
    //            State state = (State)_state[p.Handle];

    //            if (!state.Step.HasValue)
    //            {
    //                state.Step = 0;

    //                state.S = state.S.DisposeAndReplace(zeros_like(p.flatten().slice(0, 0, -1, sliceP)).detach(), true);

    //                Tensor pAny = p.any();
    //                bool[] pAnyData = pAny.ToArray<bool>();

    //                if (pAnyData.Any())
    //                    state.P0 = state.P0.DisposeAndReplace(p.flatten().slice(0, 0, -1, sliceP).detach().clone(), true);
    //                else
    //                    // All values are zero, so save VRAM with a zero-tensor
    //                    state.P0 = state.P0.DisposeAndReplace(tensor(0, device: p.device, dtype: p.dtype), true);

    //                // Exponential moving average of gradient values
    //                if (beta1.item<float>() > 0)
    //                    state.ExpAvg = state.ExpAvg.DisposeAndReplace(zeros_like(p).detach(), true);

    //                // Exponential moving average of squared gradient values
    //                state.ExpAvgSq = state.ExpAvgSq.DisposeAndReplace(zeros_like(p).detach(), true);
    //            }

    //            // Skyfly: ExpAvgSq and P0 get their value set in the if scope at line 136~ if (!state.Step.HasValue)
    //            Tensor expAvgSq = state.ExpAvgSq!;
    //            Tensor s = state.Step.Value;
    //            Tensor p0 = state.P0!;
    //            Tensor expAvg;

    //            if (groupLr > 0)
    //            {
    //                // we use d / d0 instead of just d to avoid getting values that are too small

    //                Tensor slicedGrad = grad.flatten().slice(0, 0, -1, sliceP);
    //                deltaNumerator += (d / d0) * dlr * dot(slicedGrad, p0 - p0.flatten().slice(0, 0, -1, sliceP));

    //                //# Adam EMA updates
    //                if (beta1.item<float>() > 0)
    //                {
    //                    var x = d * (1 - beta1);
    //                    expAvg = state.ExpAvg!;
    //                    expAvg!.mul_(beta1).add_(grad, (d * (1 - beta1)).ToScalar());
    //                }

    //                expAvgSq.mul_(beta2).addcmul_(grad, grad, (d * d * (1 - beta2)).ToScalar());

    //                if (safeguardWarmup)
    //                    s.mul_(beta3).add_(slicedGrad, (d / d0 * d).ToScalar());
    //                else
    //                    s.mul_(beta3).add_(slicedGrad, (d / d0 * dlr).ToScalar());

    //                dDenom += s.abs().sum().item<float>();
    //            }
    //        }

    //        Tensor dHat = d;
    //        //Tensor distTensor;
    //        Tensor? globalDNumerator = null;
    //        Tensor? globalDDenom = null;

    //        if (lr > 0)
    //        {
    //            // Unless we have an fsdp implementation, we will never run this
    //            //if (fsdpInUse)
    //            //{
    //            //    distTensor = tensor([deltaNumerator, dDenom], device: device(DeviceType.CUDA));
    //            //    dist.all_reduce(dist_tensor, op = dist.ReduceOp.SUM);
    //            //    globalDNumerator = dNumerator + distTensor[0];
    //            //    globalDDenom = distTensor[1];
    //            //}
    //            //else
    //            //{
    //            //    globalDNumerator = dNumerator + deltaNumerator;
    //            //    globalDDenom = dDenom;
    //            //}

    //            globalDNumerator = dNumerator + deltaNumerator;
    //            globalDDenom = dDenom;
    //            dHat = dCoef * globalDDenom / globalDDenom;

    //            // Skyfly: standard implementation is no null check, just equal
    //            // Skyfly: the thing is, D could be null here, so let's make sure to check it
    //            if (group.D is not null && d.equal(group.D).item<bool>())
    //            {
    //                d = max(d, dHat);
    //            }

    //            dMax = max(dMax, dHat);
    //            d = min(dMax, d * growthRate);
    //        }

    //        foreach (ParamGroup pgroup in _parameter_groups)
    //        {
    //            pgroup.DNumerator = pgroup.DNumerator.DisposeAndReplace(globalDNumerator, true);
    //            pgroup.DDenom = pgroup.DDenom.DisposeAndReplace(globalDNumerator, true);
    //            pgroup.D = pgroup.D.DisposeAndReplace(globalDNumerator, true);
    //            pgroup.DMax = pgroup.DMax.DisposeAndReplace(globalDNumerator, true);
    //            pgroup.DHat = pgroup.DHat.DisposeAndReplace(globalDNumerator, true);

    //            foreach (Parameter p in pgroup.Parameters)
    //            {
    //                if (p.grad is null)
    //                    continue;

    //                Tensor grad = p.grad;

    //                State state = (State)_state[p.Handle];

    //                // Skyfly: ExpAvgSq and ExpAvg get their value set in the if scope at line 136~ if (!state.Step.HasValue)

    //                Tensor expAvgSq = state.ExpAvgSq!;

    //                state.Step++;

    //                Tensor denom = expAvgSq.sqrt().add(d * eps);

    //                // Apply weight decay (decoupled variant)
    //                if (decay != 0 && decouple)
    //                    p.add_(p, (decay * dlr).ToScalar());

    //                // Take step
    //                if (beta1.item<float>() > 0)
    //                {
    //                    Tensor expAvg = state.ExpAvg!;
    //                    p.addcdiv_(expAvg, denom, dlr.ToScalar());
    //                }
    //                else
    //                {
    //                    p.addcdiv_(grad, denom, (dlr * d).ToScalar());
    //                }
    //            }

    //            pgroup.K = k.item<float>() + 1;
    //        }
    //    }, closure);
    //}


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
            Step = null;

            S?.Dispose();
            S = null;

            P0?.Dispose();
            P0 = null;

            ExpAvg?.Dispose();
            ExpAvg = null;

            ExpAvgSq?.Dispose();
            ExpAvgSq = null;

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

        public (float, float) BetasFloat
        {
            get => (Options.Beta1, Options.Beta2);
            set
            {
                Options.Beta1 = value.Item1;
                Options.Beta2 = value.Item2;
            }
        }
        public float? Beta3 { get; set; }
        public float DNumerator { get; set; }
        public float DDenom { get; set; }
        public float D { get; set; }
        public float D0 { get; set; }
        public float DCoef { get; set; }
        public float DMax { get; set; }
        public float DHat { get; set; }
        public bool Decouple { get; set; }
        public bool SafeguardWarmup { get; set; }
        public float GrowthRate { get; set; }
        public float WeightDecay { get; set; }
        public float K { get; set; }
        public float EPS { get; set; } 
        public bool UseBiasCorrection { get; set; }
        public long SliceP { get; set; }

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