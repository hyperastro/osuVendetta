using OsuParsers.Enums;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays.Data;
using System.Diagnostics;

namespace osuVendetta.Core.Replays;

public class ReplayProcessor : IReplayProcessor
{
    readonly IAntiCheatModel _model;

    public ReplayProcessor(IAntiCheatModel anticheatModel)
    {
        _model = anticheatModel;
    }

    public ReplayValidationResult IsValidReplay(Replay replay)
    {
        if (replay.ReplayFrames.Count == 0)
            return new ReplayValidationResult(false, $"Replay has no frames");

        switch (replay.Ruleset)
        {
            case Ruleset.Standard:
                return new ReplayValidationResult(true, null);

            default:
                return new ReplayValidationResult(false, $"Unsupported Ruleset ({replay.Ruleset})");
        }
    }

    public async Task<ReplayTokens> CreateTokensFromFramesAsync(string replayName, List<ReplayFrame> frames, bool runInParallel)
    {
        // include the first chunk being 1000 frames instead of 500 (-size, +1)
        int totalChunks = (int)Math.Ceiling((frames.Count - _model.Config.StepsPerChunk) / (float)_model.Config.StepOverlay) + 1;

        if (totalChunks <= 0)
            Debugger.Break();

        float[] inputs = new float[totalChunks * _model.Config.TotalFeatureSizePerChunk];

        await Task.Run(() =>
        {
            ScalerValues mean = _model.Config.ScalerMean;
            ScalerValues std = _model.Config.ScalerStd;

            if (runInParallel)
            {
                Parallel.For(0, frames.Count, index =>
                {
                    ProcessFrame(index, frames, inputs, ref mean, ref std);
                });
            }
            else
            {
                for (int i = 0; i < frames.Count; i++)
                    ProcessFrame(i, frames, inputs, ref mean, ref std);
            }

        });

        // TODO: temporary fix, replace method later
        int batchSize = (int)Math.Ceiling((float)inputs.Length / _model.Config.TotalFeatureSizePerChunk);
        float[,,] input = new float[batchSize, _model.Config.StepsPerChunk, _model.Config.FeaturesPerStep];

        int inputIdx = 0;
        for (int idx0 = 0; idx0 < batchSize; idx0++)
        {
            for (int idx1 = 0; idx1 < _model.Config.StepsPerChunk; idx1++)
            {
                for (int idx2 = 0; idx2 < _model.Config.FeaturesPerStep; idx2++)
                {
                    input[idx0, idx1, idx2] = inputs[inputIdx++];
                }
            }
        }

        return new ReplayTokens
        {
            Tokens = input,
            ReplayName = replayName
        };
    }

    void ProcessFrame(int index, List<ReplayFrame> frames, float[] inputs,
        ref ScalerValues scalerMean, ref ScalerValues scalerStd)
    {
        ReplayFrame currentFrame = frames[index];
        float deltaX = currentFrame.X;
        float deltaY = currentFrame.Y;

        if (index > 0)
        {
            ReplayFrame lastFrame = frames[index - 1];
            deltaX -= lastFrame.X;
            deltaY -= lastFrame.Y;
        }

        int indexMain;
        int indexOverflow = -1;

        if (index < _model.Config.StepOverlay)
        {
            indexMain = index;
        }
        else if (index < _model.Config.StepsPerChunk)
        {
            indexMain = index;
            indexOverflow = index + _model.Config.StepOverlay;
        }
        else
        {
            int chunkIndex = (index - _model.Config.StepsPerChunk) / _model.Config.StepOverlay + 1;
            int chunkOffset = index % _model.Config.StepOverlay;

            indexMain = chunkIndex * _model.Config.StepsPerChunk + chunkOffset + _model.Config.StepOverlay;
            indexOverflow = indexMain + _model.Config.StepOverlay;
        }

        indexMain *= _model.Config.FeaturesPerStep;
        indexOverflow *= _model.Config.FeaturesPerStep;

        inputs[indexMain + 0] = Normalize(currentFrame.TimeDiff, scalerMean.DimensionDeltaTime, scalerStd.DimensionDeltaTime);
        inputs[indexMain + 1] = Normalize(currentFrame.X, scalerMean.DimensionX, scalerStd.DimensionX);
        inputs[indexMain + 2] = Normalize(currentFrame.Y, scalerMean.DimensionY, scalerStd.DimensionY);
        inputs[indexMain + 3] = Normalize(deltaX, scalerMean.DimensionDeltaX, scalerStd.DimensionDeltaX);
        inputs[indexMain + 4] = Normalize(deltaY, scalerMean.DimensionDeltaY, scalerStd.DimensionDeltaY);
        inputs[indexMain + 5] = GetKeyValue(currentFrame.StandardKeys);

        if (indexOverflow > -1 && indexOverflow < inputs.Length)
        {
            inputs[indexOverflow + 0] = inputs[indexMain + 0];
            inputs[indexOverflow + 1] = inputs[indexMain + 1];
            inputs[indexOverflow + 2] = inputs[indexMain + 2];
            inputs[indexOverflow + 3] = inputs[indexMain + 3];
            inputs[indexOverflow + 4] = inputs[indexMain + 4];
            inputs[indexOverflow + 5] = inputs[indexMain + 5];
        }
    }

    static float Normalize(double value, double mean, double stdDev)
    {
        return (float)((value - mean) / stdDev);
    }

    static float GetKeyValue(StandardKeys keys)
    {
        switch (keys)
        {
            case StandardKeys.M1:
                return 0f;

            case StandardKeys.M1 | StandardKeys.M2:
                return 1f;

            case StandardKeys.M2:
                return 2f;

            default:
                return 3f;
        }
    }
}