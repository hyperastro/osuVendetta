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

        // Convert logits to probabilities using a softmax function
        var probabilities = Softmax(probabilityRelaxTotal, probabilityNormalTotal);

        string message = $"Normal Probability: {probabilities.normalProb:F4}/Relax Probability: {probabilities.relaxProb:F4}";

        if (probabilityRelaxTotal < probabilityNormalTotal)
            return AntiCheatResult.Relax(message);
        else
            return AntiCheatResult.Normal(message);
    }

    // Softmax function
    private (float relaxProb, float normalProb) Softmax(float logitRelax, float logitNormal)
    {
        float maxLogit = Math.Max(logitRelax, logitNormal);
        float expRelax = MathF.Exp(logitRelax - maxLogit);
        float expNormal = MathF.Exp(logitNormal - maxLogit);
        float sumExp = expRelax + expNormal;

        return ((expNormal / sumExp)*100, (expRelax / sumExp)*100);
    }

    protected abstract void Unload();

    protected abstract Task<ProbabilityResult> RunModelAsync(Memory<float> input, long[] shape);
}
