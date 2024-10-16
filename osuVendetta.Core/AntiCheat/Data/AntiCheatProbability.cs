using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.AntiCheat.Data;

/// <summary>
/// Probability if a play is cheated
/// </summary>
public struct AntiCheatProbability
{
    /// <summary>
    /// Probability of it being cheated with relax
    /// </summary>
    public required float Relax;

    /// <summary>
    /// Probability of it being legit
    /// </summary>
    public required float Normal;

    public AntiCheatProbability(float relax, float normal) : this()
    {
        Relax = relax;
        Normal = normal;
    }

    /// <summary>
    /// <inheritdoc cref="Softmax(float, float)"/>
    /// </summary>
    /// <param name="logits">Max Logit Values</param>
    /// <param name="totalChunks">Chunks used to create max logits</param>
    /// <returns>A <see cref="ProbabilityResult"/> contains the probabilities for relax and normal.</returns>
    public static AntiCheatProbability FromLogits(ConcurrentQueue<Logit> logits, int totalChunks)
    {
        float probabilityRelaxTotal = 0;
        float probabilityNormalTotal = 0;

        while (logits.TryDequeue(out Logit logit))
        {
            probabilityRelaxTotal += logit.Relax;
            probabilityNormalTotal += logit.Normal;
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
    static AntiCheatProbability Softmax(float logitNormal, float logitRelax)
    {
        float maxLogit = Math.Max(logitRelax, logitNormal);
        float expRelax = MathF.Exp(logitRelax - maxLogit);
        float expNormal = MathF.Exp(logitNormal - maxLogit);
        float sumExp = expRelax + expNormal;

        return new AntiCheatProbability
        {
            Relax = (expRelax / sumExp) * 100,
            Normal = (expNormal / sumExp) * 100
        };
    }
}
