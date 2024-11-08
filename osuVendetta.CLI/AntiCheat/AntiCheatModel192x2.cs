using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using Microsoft.ML.OnnxRuntime;
using osuVendetta.CoreLib.AntiCheat.Data;
using osuVendetta.Core.Replays.Data;
using System.Security.Cryptography;
using System.Text;
using System.Dynamic;

namespace osuVendetta.CLI.AntiCheat;

public class AntiCheatModel192x2 : IAntiCheatModel
{
    RunOptions? _runOptions;
    InferenceSession? _inferenceSession;

    public Task LoadAsync(AntiCheatModelLoadArgs loadArgs)
    {
        if (_inferenceSession is not null)
            throw new InvalidOperationException("Unload the model first before loading a new one");

        SessionOptions options = new SessionOptions
        {

        };

        _runOptions = new RunOptions();
        _inferenceSession = new InferenceSession(loadArgs.ModelBytes, options);

        return Task.CompletedTask;
    }

    public Task<Logit> Run(ModelInput input)
    {
        ArgumentNullException.ThrowIfNull(_runOptions);
        ArgumentNullException.ThrowIfNull(_inferenceSession);

        using OrtValue value = OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance,
            input.Data,
            input.DataShape);

        Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>
        {
            { "input", value }
        };

        using IDisposableReadOnlyCollection<OrtValue> output = _inferenceSession.Run(_runOptions, inputs, _inferenceSession.OutputNames);
        ReadOnlySpan<float> outputData = output[0].GetTensorDataAsSpan<float>();

        return Task.FromResult(new Logit
        {
            Relax = outputData[0],
            Normal = outputData[1]
        });
    }

    public Task UnloadAsync()
    {
        _inferenceSession?.Dispose();
        _inferenceSession = null;

        _runOptions?.Dispose();
        _runOptions = null;

        return Task.CompletedTask;
    }

    public static AntiCheatConfig CreateDefaultConfigForModel()
    {
        return new AntiCheatConfig
        {
            Version = new ModelVersion
            {
                DisplayText = "Onnx 192x2 Alpha Model",
                Major = 1,
                Minor = 0
            },
            ScalerMean = new Scaler
            {
                DimensionDeltaTime = 7.03284752e+00,
                DimensionDeltaX = 2.09958789e-03,
                DimensionDeltaY = 2.68233697e-02,
            },
            ScalerStd = new Scaler
            {
                DimensionDeltaTime = 562.68982467,
                DimensionDeltaX = 27.54802019,
                DimensionDeltaY = 27.51032391,
            },
            BatchCount = 1,
            FeaturesPerStep = 6,
            StepOverlay = 500,
            StepsPerChunk = 1000,
        };
    }
}
