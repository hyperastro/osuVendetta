using OsuParsers.Enums;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.Replays.Data;
using osuVendetta.CoreLib.AntiCheat.Data;

namespace osuVendetta.Core.Replays;


public class ReplayProcessor : IReplayProcessor
{
    readonly AntiCheatConfig _config;

    public ReplayProcessor(AntiCheatConfig config)
    {
        _config = config;
    }

    public ReplayValidationResult IsValidReplay(Replay replay)
    {
        switch (replay.Ruleset)
        {
            case Ruleset.Standard:
                return new ReplayValidationResult(true, null);

            default:
                return new ReplayValidationResult(false, $"Unsupported Ruleset ({replay.Ruleset})");
        }
    }

    public async Task<ReplayTokens> CreateTokensFromFramesAsync(List<ReplayFrame> frames, bool runInParallel)
    {
        // include the first chunk being 1000 frames instead of 500 (-size, +1)
        int totalChunks = (int)Math.Ceiling((frames.Count - _config.StepsPerChunk) / (float)_config.StepOverlay) + 1;
        float[] inputs = new float[totalChunks * _config.TotalFeatureSizePerChunk];

        await Task.Run(() =>
        {
            ScalerValues mean = _config.ScalerMean;
            ScalerValues std = _config.ScalerStd;

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

        long[] dimensions = CreateModelDimensions(inputs);
        return new ReplayTokens(inputs, dimensions);
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

        if (index < _config.StepOverlay)
        {
            indexMain = index;
        }
        else if (index < _config.StepsPerChunk)
        {
            indexMain = index;
            indexOverflow = index + _config.StepOverlay;
        }
        else
        {
            int chunkIndex = (index - _config.StepsPerChunk) / _config.StepOverlay + 1;
            int chunkOffset = index % _config.StepOverlay;

            indexMain = chunkIndex * _config.StepsPerChunk + chunkOffset + _config.StepOverlay;
            indexOverflow = indexMain + _config.StepOverlay;
        }

        indexMain *= _config.FeaturesPerStep;
        indexOverflow *= _config.FeaturesPerStep;

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

    long[] CreateModelDimensions(float[] modelInput)
    {
        ArgumentNullException.ThrowIfNull(_config);

        return [
            _config.BatchCount,
            _config.StepsPerChunk,
            _config.FeaturesPerStep
        ];
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