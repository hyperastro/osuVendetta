using OsuParsers.Enums;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.AntiCheat.Data;

namespace osuVendetta.Core.AntiCheat;

public record class RunModelArgs(
    AntiCheatModelProviderArgs ModelProviderArgs,
    InputArgs InputArgs
);

public record struct MaxValueIndex(int MaxIndex, float MaxValue);

public interface IAntiCheatService
{
    Task<AntiCheatResult> RunModelAsync(RunModelArgs args);
    string? ValidateReplay(Replay replay);
    float[] ProcessReplayTokens(List<ReplayFrame> frames);
    long[] CreateModelDimensionsFor(float[] input);
}

public sealed class AntiCheatService : IAntiCheatService
{
    const int MAX_TIME_STEPS = 8000;
    const int FEATURES_PER_ROW = 6;
    const int MAX_INPUTS = MAX_TIME_STEPS * FEATURES_PER_ROW;

    readonly IAntiCheatModelProvider _modelProvider;

    public AntiCheatService(IAntiCheatModelProvider modelProvider)
    {
        _modelProvider = modelProvider;
    }

    public long[] CreateModelDimensionsFor(float[] input)
    {
        return new long[]
        {
            1, // amount of inputs
            MAX_TIME_STEPS, // max dimensions
            FEATURES_PER_ROW // features per dimension
        };
    }

    public float[] ProcessReplayTokens(List<ReplayFrame> frames)
    {
        float[] inputs = new float[Math.Min(frames.Count * FEATURES_PER_ROW, MAX_INPUTS)];
        int framesToProcess = Math.Min(frames.Count, MAX_TIME_STEPS);

        float lastX = frames[0].X;
        float lastY = frames[0].Y;
        int indexInputs = 0;

        for (int i = 0; i < framesToProcess; i++)
        {
            ReplayFrame frame = frames[i];

            inputs[indexInputs++] = frame.TimeDiff;
            inputs[indexInputs++] = frame.X;
            inputs[indexInputs++] = frame.Y;
            inputs[indexInputs++] = frame.X - Interlocked.Exchange(ref lastX, frame.X);
            inputs[indexInputs++] = frame.Y - Interlocked.Exchange(ref lastY, frame.Y);
            inputs[indexInputs++] = GetKeyValue(frame.StandardKeys);
        }

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
