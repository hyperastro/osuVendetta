using osuVendetta.Core.AntiCheat.Data;
using System.Collections.Concurrent;

namespace osuVendetta.Core.AntiCheat;

public record struct ProbabilityResult(float ProbabilityRelax, float ProbabilityNormal);

public abstract class AntiCheatModel : IAntiCheatModel
{

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
        ConcurrentQueue<ProbabilityResult> resultLogits = new ConcurrentQueue<ProbabilityResult>();

        await Parallel.ForAsync(0, totalChunks, async (chunkIndex, _) =>
        {
            int start = chunkIndex * AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK;

            Memory<float> inputMemory = new Memory<float>(args.InputData, start, AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK);
            ProbabilityResult probability = await RunModelAsync(inputMemory, args.Dimensions);

            resultLogits.Enqueue(probability);
        });

        ProbabilityResult probabilities = ProcessLogitsToProbability(resultLogits, totalChunks);

        AntiCheatResult result;

        if (probabilities.ProbabilityRelax > probabilities.ProbabilityNormal)
            result = AntiCheatResult.Relax();
        else
            result = AntiCheatResult.Normal();

        result.ProbabilityResult = probabilities;

        return result;
    }

    /// <summary>
    /// <inheritdoc cref="Softmax(float, float)"/>
    /// </summary>
    /// <param name="logits">Max Logit Values</param>
    /// <param name="totalChunks">Chunks used to create max logits</param>
    /// <returns>A <see cref="ProbabilityResult"/> contains the probabilities for relax and normal.</returns>
    ProbabilityResult ProcessLogitsToProbability(ConcurrentQueue<ProbabilityResult> logits, int totalChunks)
    {
        float probabilityRelaxTotal = 0;
        float probabilityNormalTotal = 0;

        while (logits.TryDequeue(out ProbabilityResult probabilityResult))
        {
            probabilityRelaxTotal += probabilityResult.ProbabilityRelax;
            probabilityNormalTotal += probabilityResult.ProbabilityNormal;
        }

        probabilityRelaxTotal /= totalChunks;
        probabilityNormalTotal /= totalChunks;

        return Softmax(probabilityRelaxTotal, probabilityNormalTotal);
    }

    /// <summary>
    /// Convert logits to probabilities using a softmax function.
    /// </summary>
    /// <param name="logitRelax">Logit value for Relax.</param>
    /// <param name="logitNormal">Logit value for Normal.</param>
    /// <returns>A <see cref="ProbabilityResult"/> contains the probabilities for relax and normal.</returns>
    ProbabilityResult Softmax(float logitNormal, float logitRelax)
    {
        float maxLogit = Math.Max(logitRelax, logitNormal);
        float expRelax = MathF.Exp(logitRelax - maxLogit);
        float expNormal = MathF.Exp(logitNormal - maxLogit);
        float sumExp = expRelax + expNormal;

        return new ProbabilityResult(
            ProbabilityRelax: (expRelax / sumExp) * 100,
            ProbabilityNormal: (expNormal / sumExp) * 100
        );
    }

    protected abstract void Unload();

    protected abstract Task<ProbabilityResult> RunModelAsync(Memory<float> input, long[] shape);
}
