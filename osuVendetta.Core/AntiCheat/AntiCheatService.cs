using OsuParsers.Decoders;
using OsuParsers.Enums;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.AntiCheat.Data;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;

namespace osuVendetta.Core.AntiCheat;

public record class RunModelArgs(
    AntiCheatModelProviderArgs ModelProviderArgs,
    InputArgs InputArgs
);

public record struct MaxValueIndex(
    int MaxIndex, 
    float MaxValue
);

public interface IAntiCheatService
{
    Task<AntiCheatResult> ProcessReplayAsync(Stream replayStream, AntiCheatModelProviderArgs modelProviderArgs);
    Task<AntiCheatResult> ProcessReplayAsync(Replay replay, AntiCheatModelProviderArgs modelProviderArgs);
    Task<AntiCheatResult> RunModelAsync(RunModelArgs args);
    string? ValidateReplay(Replay replay);
    long[] CreateModelDimensionsFor(float[] input);
}

public sealed class AntiCheatService : IAntiCheatService
{
    const int BATCH_COUNT = 1; // 1d array
    const int MAX_TIME_STEPS = 8000;
    const int FEATURES_PER_ROW = 6;
    const int MAX_INPUTS = MAX_TIME_STEPS * FEATURES_PER_ROW;

    public const int CHUNK_SIZE = 1000;
    public const int FRAME_OVERLAY = 500;
    public const int FEATURES_PER_FRAME = 6;
    public const int TOTAL_FEATURE_SIZE_CHUNK = CHUNK_SIZE * FEATURES_PER_FRAME;

    const double MEAN_R1 = 7.03284752e+00f;
    const double MEAN_R2 = 2.09958789e-03f;
    const double MEAN_R3 = 2.68233697e-02f;

    const double STD_R1 = 562.68982467f;
    const double STD_R2 = 27.54802019f;
    const double STD_R3 = 27.51032391f;


    readonly IAntiCheatModelProvider _modelProvider;

    public AntiCheatService(IAntiCheatModelProvider modelProvider)
    {
        _modelProvider = modelProvider;
    }

    public async Task<AntiCheatResult> ProcessReplayAsync(Stream replayStream, AntiCheatModelProviderArgs modelProviderArgs)
    {
        Replay replay = ReplayDecoder.Decode(replayStream);

        return await ProcessReplayAsync(replay, modelProviderArgs);
    }

    public async Task<AntiCheatResult> ProcessReplayAsync(Replay replay, AntiCheatModelProviderArgs modelProviderArgs)
    {
        string? replayValidation = ValidateReplay(replay);

        if (!string.IsNullOrEmpty(replayValidation))
            return AntiCheatResult.Invalid(replayValidation);

        float[] replayTokens = ProcessReplayTokensNew(replay.ReplayFrames);
        long[] dimensions = CreateModelDimensionsFor(replayTokens);

        RunModelArgs modelArgs = new RunModelArgs(modelProviderArgs, new InputArgs(replayTokens, dimensions));

        AntiCheatResult result = await RunModelAsync(modelArgs);
        result.Metadata = new AntiCheatResultMetadata(replay.PlayerName);

        return result;
    }

    public long[] CreateModelDimensionsFor(float[] input)
    {
        return new long[]
        {
            BATCH_COUNT,
            CHUNK_SIZE,
            FEATURES_PER_ROW
        };
    }

    public float[] ProcessReplayTokensNew(List<ReplayFrame> frames)
    {
        // include the first chunk being 1000 frames instead of 500 (-size, +1)
        int totalChunks = (int)Math.Ceiling((frames.Count - CHUNK_SIZE) / (float)FRAME_OVERLAY) + 1;
        float[] inputs = new float[totalChunks * CHUNK_SIZE * FEATURES_PER_FRAME];

        Parallel.For(0, frames.Count, i =>
        {
            ReplayFrame frame = frames[i];

            float deltaX = frame.X;
            float deltaY = frame.Y;

            if (i > 0)
            {
                ReplayFrame lastFrame = frames[i - 1];
                deltaX -= lastFrame.X;
                deltaY -= lastFrame.Y;
            }

            int indexMain;
            int indexOverflow = -1;

            if (i < FRAME_OVERLAY)
            {
                indexMain = i;
            }
            else if (i < CHUNK_SIZE)
            {
                indexMain = i;
                indexOverflow = i + FRAME_OVERLAY;
            }
            else
            {
                int chunkIndex = (i - CHUNK_SIZE) / FRAME_OVERLAY + 1;
                int chunkOffset = i % FRAME_OVERLAY;
                 
                indexMain = chunkIndex * CHUNK_SIZE + chunkOffset + FRAME_OVERLAY;
                indexOverflow = indexMain + FRAME_OVERLAY;
            }

            indexMain *= FEATURES_PER_FRAME;
            indexOverflow *= FEATURES_PER_FRAME;

            inputs[indexMain + 0] = Normalize(frame.TimeDiff, MEAN_R1, STD_R1);
            inputs[indexMain + 1] = frame.X;
            inputs[indexMain + 2] = frame.Y;
            inputs[indexMain + 3] = Normalize(deltaX, MEAN_R2, STD_R2);
            inputs[indexMain + 4] = Normalize(deltaY, MEAN_R3, STD_R3);
            inputs[indexMain + 5] = GetKeyValue(frame.StandardKeys);

            if (indexOverflow > -1 && indexOverflow < inputs.Length)
            {
                inputs[indexOverflow + 0] = inputs[indexMain + 0];
                inputs[indexOverflow + 1] = inputs[indexMain + 1];
                inputs[indexOverflow + 2] = inputs[indexMain + 2];
                inputs[indexOverflow + 3] = inputs[indexMain + 3];
                inputs[indexOverflow + 4] = inputs[indexMain + 4];
                inputs[indexOverflow + 5] = inputs[indexMain + 5];
            }
        });

        return inputs;
    }

    public async Task<AntiCheatResult> RunModelAsync(RunModelArgs args)
    {
        using IAntiCheatModel model = await _modelProvider.LoadModelAsync(args.ModelProviderArgs);

        return await model.RunModelAsync(args.InputArgs);
    }

    /// <summary>
    /// Validates the replay, if the replay is invalid returns the error message
    /// </summary>
    /// <param name="replay"></param>
    /// <returns></returns>
    public string? ValidateReplay(Replay replay)
    {
        switch (replay.Ruleset)
        {
            case Ruleset.Standard:
                break;

            default:
                return $"Unsupported Ruleset ({replay.Ruleset})";
        }

        // Replay is valid
        return null;
    }

    float Normalize(double value, double mean, double stdDev)
    {
        return (float)((value - mean) / stdDev);
    }

    float GetKeyValue(StandardKeys keys)
    {
        // m1 = 0f, m1m2 = 1f, m2 = 2f, else = 3f

        if (keys.HasFlag(StandardKeys.M1))
        {
            if (keys.HasFlag(StandardKeys.M2))
                return 1f;
            else
                return 0f;
        }
        else if (keys.HasFlag(StandardKeys.M2))
        {
            return 2f;
        }
        else
        {
            return 3f;
        }
    }
}
