using OsuParsers.Enums;
using OsuParsers.Replays;
using osuVendetta.Core.AntiCheat.Data;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using osuVendetta.CoreLib.AntiCheat.Data;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.AntiCheat;

public class AntiCheatRunner
{
    readonly ReplayProcessor _replayProcessor;
    readonly IAntiCheatModel _antiCheatModel;
    readonly AntiCheatConfig _config;

    public AntiCheatRunner(IAntiCheatModel antiCheatModel, AntiCheatConfig config)
    {
        _replayProcessor = new ReplayProcessor(config);
        _antiCheatModel = antiCheatModel;
        _config = config;
    }

    public async Task<BaseAntiCheatResult> ProcessReplayAsync(Replay replay, bool runInParallel)
    {
        try
        {
            ReplayValidationResult validation = _replayProcessor.IsValidReplay(replay);

            if (!validation.IsValid)
            {
                return new AntiCheatNotSupportedRulesetError
                {
                    RequestedRuleset = replay.Ruleset.ToString(),
                    SupportedRulesets = Ruleset.Standard.ToString()
                };
            }

            ReplayTokens tokens = await _replayProcessor.CreateTokensFromFramesAsync(replay.ReplayFrames, runInParallel);
            return await RunModel(tokens, runInParallel);
        }
        catch (Exception ex)
        {
            return new AntiCheatUnhandledError
            {
                Error = ex.ToString()
            };
        }
    }

    async Task<BaseAntiCheatResult> RunModel(ReplayTokens tokens, bool runInParallel)
    {
        int totalChunks = (int)Math.Ceiling(tokens.Tokens.Length / (double)_config.TotalFeatureSizePerChunk);
        ConcurrentQueue<Logit> resultLogits = new ConcurrentQueue<Logit>();

        if (runInParallel)
        {
            await Parallel.ForAsync(0, totalChunks, async (chunkIndex, _) 
                => await RunOnChunk(chunkIndex, tokens, resultLogits));
        }
        else
        {
            for (int chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++)
                await RunOnChunk(chunkIndex, tokens, resultLogits);
        }

        return new AntiCheatResult
        {
            CheatProbability = ProcessLogitsToProbability(resultLogits, totalChunks)
        };
    }

    async Task RunOnChunk(int chunkIndex, ReplayTokens tokens, ConcurrentQueue<Logit> resultLogits)
    {
        int start = chunkIndex * _config.TotalFeatureSizePerChunk;

        Logit logit = await _antiCheatModel.Run(new ModelInput
        {
            Data = new Memory<float>(tokens.Tokens, start, _config.TotalFeatureSizePerChunk),
            DataShape = tokens.ModelDimensions
        });

        resultLogits.Enqueue(logit);
    }



    /// <summary>
    /// <inheritdoc cref="Softmax(float, float)"/>
    /// </summary>
    /// <param name="logits">Max Logit Values</param>
    /// <param name="totalChunks">Chunks used to create max logits</param>
    /// <returns>A <see cref="ProbabilityResult"/> contains the probabilities for relax and normal.</returns>
    AntiCheatProbability ProcessLogitsToProbability(ConcurrentQueue<Logit> logits, int totalChunks)
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

        return Softmax(probabilityNormalTotal, probabilityRelaxTotal);
    }

    /// <summary>
    /// Convert logits to probabilities using a softmax function.
    /// </summary>
    /// <param name="logitRelax">Logit value for Relax.</param>
    /// <param name="logitNormal">Logit value for Normal.</param>
    /// <returns>A <see cref="ProbabilityResult"/> contains the probabilities for relax and normal.</returns>
    AntiCheatProbability Softmax(float logitNormal, float logitRelax)
    {
        float maxLogit = Math.Max(logitRelax, logitNormal);
        float expRelax = MathF.Exp(logitRelax - maxLogit);
        float expNormal = MathF.Exp(logitNormal - maxLogit);
        float sumExp = expRelax + expNormal;

        return new AntiCheatProbability
        {
            Relax = (expRelax / sumExp) * 100,
            Normal = (expNormal / sumExp) * 100,
        };
    }
}
