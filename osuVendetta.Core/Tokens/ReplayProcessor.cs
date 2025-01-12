﻿using OsuParsers.Decoders;
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

    public ReplayValidationResult IsValidReplay(Stream replayData)
    {
        Replay replay = ReplayDecoder.Decode(replayData);

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

    public ReplayTokens CreateTokensParallel(Stream replayData)
    {
        Replay replay = ReplayDecoder.Decode(replayData);
        List<ReplayFrame> frames = replay.ReplayFrames;

        // include the first chunk being 1000 frames instead of 500 (-size, +1)
        int totalChunks = frames.Count / _model.Config.StepsPerChunk;
        //int totalChunks = (int)Math.Ceiling((frames.Count - _model.Config.StepsPerChunk) / (float)_model.Config.StepOverlay) + 1;
        float[] inputs = new float[totalChunks * _model.Config.TotalFeatureSizePerChunk];

        ScalerValues mean = _model.Config.ScalerMean;
        ScalerValues std = _model.Config.ScalerStd;
        Parallel.For(0, frames.Count, index =>
        {
            ProcessFrame(index, frames, inputs, ref mean, ref std);
        });

        return new ReplayTokens
        {
            Tokens = inputs,
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

        int indexMain = index;
        //int indexMain;
        //int indexOverflow = -1;

        //if (index < _model.Config.StepOverlay)
        //{
        //    indexMain = index;
        //}
        //else if (index < _model.Config.StepsPerChunk)
        //{
        //    indexMain = index;
        //    indexOverflow = index + _model.Config.StepOverlay;
        //}
        //else
        //{
        //    int chunkIndex = (index - _model.Config.StepsPerChunk) / _model.Config.StepOverlay + 1;
        //    int chunkOffset = index % _model.Config.StepOverlay;

        //    indexMain = chunkIndex * _model.Config.StepsPerChunk + chunkOffset + _model.Config.StepOverlay;
        //    indexOverflow = indexMain + _model.Config.StepOverlay;
        //}

        indexMain *= _model.Config.FeaturesPerStep;
        //indexOverflow *= _model.Config.FeaturesPerStep;

        inputs[indexMain + 0] = Normalize(currentFrame.TimeDiff, scalerMean.DimensionDeltaTime, scalerStd.DimensionDeltaTime);
        inputs[indexMain + 1] = Normalize(currentFrame.X, scalerMean.DimensionX, scalerStd.DimensionX);
        inputs[indexMain + 2] = Normalize(currentFrame.Y, scalerMean.DimensionY, scalerStd.DimensionY);
        inputs[indexMain + 3] = Normalize(deltaX, scalerMean.DimensionDeltaX, scalerStd.DimensionDeltaX);
        inputs[indexMain + 4] = Normalize(deltaY, scalerMean.DimensionDeltaY, scalerStd.DimensionDeltaY);
        inputs[indexMain + 5] = GetKeyValue(currentFrame.StandardKeys);

        //if (indexOverflow > -1 && indexOverflow < inputs.Length)
        //{
        //    inputs[indexOverflow + 0] = inputs[indexMain + 0];
        //    inputs[indexOverflow + 1] = inputs[indexMain + 1];
        //    inputs[indexOverflow + 2] = inputs[indexMain + 2];
        //    inputs[indexOverflow + 3] = inputs[indexMain + 3];
        //    inputs[indexOverflow + 4] = inputs[indexMain + 4];
        //    inputs[indexOverflow + 5] = inputs[indexMain + 5];
        //}
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