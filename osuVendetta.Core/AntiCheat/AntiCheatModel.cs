using osuVendetta.Core.AntiCheat.Data;
using System.Collections.Concurrent;

namespace osuVendetta.Core.AntiCheat;

public abstract class AntiCheatModel : IAntiCheatModel
{
    protected record struct ProbabilityResult(float ProbabilityRelax, float ProbabilityNormal);

    public bool IsDisposed { get; private set; }
    public string ModelPath { get; private set; }

    public AntiCheatModel(string modelPath)
    {
        ModelPath = modelPath;
    }

    public void Dispose()
    {
        if (IsDisposed)
            return;

        IsDisposed = true;
        Unload();
    }

    public abstract Task LoadAsync();
    //{
    //    _session?.Dispose();
    //    _session = null;
    //}

    //{
    //int start = chunkIndex * AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK;

    //using OrtValue value = OrtValue.CreateTensorValueFromMemory(
    //    OrtMemoryInfo.DefaultInstance,
    //    new Memory<float>(args.InputData, start, AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK),
    //    args.Dimensions);

    //Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>
    //        {
    //            { "input", value }
    //        };

    //using IDisposableReadOnlyCollection<OrtValue> output = _session.Run(_runOptions, inputs, _session.OutputNames);
    //ReadOnlySpan<float> outputData = output[0].GetTensorDataAsSpan<float>();

    //probabliltyRelaxTotal += outputData[0];
    //probabliltyNormalTotal += outputData[1];
    //}

    public async Task<AntiCheatResult> RunModelAsync(InputArgs args)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        int totalChunks = (int)Math.Ceiling(args.InputData.Length / (double)AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK);
        ConcurrentQueue<ProbabilityResult> results = new ConcurrentQueue<ProbabilityResult>();

        await Parallel.ForAsync(0, totalChunks, async (chunkIndex, _) =>
        {
            int start = chunkIndex * AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK;

            Memory<float> inputMemory = new Memory<float>(args.InputData, start, AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK);
            ProbabilityResult probability = await RunModelAsync(inputMemory, args.Dimensions);

            results.Enqueue(probability);
        });

        float probabilityRelaxTotal = 0;
        float probabilityNormalTotal = 0;

        while (results.TryDequeue(out ProbabilityResult probabilityResult))
        {
            probabilityRelaxTotal += probabilityResult.ProbabilityRelax;
            probabilityNormalTotal += probabilityResult.ProbabilityNormal;
        }

        probabilityRelaxTotal /= totalChunks;
        probabilityNormalTotal /= totalChunks;

        string message = $"Normal Probability: {probabilityNormalTotal}/Relax Probability: {probabilityRelaxTotal}";

        if (probabilityRelaxTotal < probabilityNormalTotal)
            return AntiCheatResult.Relax(message);
        else
            return AntiCheatResult.Normal(message);

    }

    protected abstract void Unload();

    protected abstract Task<ProbabilityResult> RunModelAsync(Memory<float> input, long[] shape);
}
