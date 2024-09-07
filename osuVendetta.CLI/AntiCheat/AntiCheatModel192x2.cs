using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using Microsoft.ML.OnnxRuntime;

namespace osuVendetta.CLI.AntiCheat;

public class AntiCheatModel192x2 : AntiCheatModel
{
    RunOptions? _runOptions;
    InferenceSession? _session;

    public AntiCheatModel192x2(string modelPath) : base(modelPath)
    {
    }

    static MaxValueIndex GetMaxValueIndex(OrtValue value)
    {
        int maxIndex = -1;
        float maxValue = float.MinValue;

        ReadOnlySpan<float> output = value.GetTensorDataAsSpan<float>();

        for (int i = 0; i < output.Length; i++)
        {
            float outputValue = output[i];

            if (outputValue > maxValue)
            {
                maxValue = outputValue;
                maxIndex = i;
            }
        }

        return new MaxValueIndex(maxIndex, maxValue);
    }

    public override async Task LoadAsync()
    {
        byte[] modelData = File.ReadAllBytes(ModelPath);

        SessionOptions options = new SessionOptions
        {

        };

        _runOptions = new RunOptions();
        _session = new InferenceSession(modelData, options);
    }

    protected override void Unload()
    {
        _session?.Dispose();
        _session = null;
    }

    protected override async Task<ProbabilityResult> RunModelAsync(Memory<float> input, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(_session);

        using OrtValue value = OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance,
            input,
            shape);

        Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>
        {
                { "input", value }
        };

        using IDisposableReadOnlyCollection<OrtValue> output = _session.Run(_runOptions, inputs, _session.OutputNames);
        return ToResult(output);
    }

    ProbabilityResult ToResult(IDisposableReadOnlyCollection<OrtValue> output)
    {
        ReadOnlySpan<float> outputData = output[0].GetTensorDataAsSpan<float>();
        return new ProbabilityResult(outputData[0], outputData[1]);
    }
}
