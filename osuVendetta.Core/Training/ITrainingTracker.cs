using osuVendetta.Core.Training;

namespace osuVendetta.Core.Training;

public enum TrainingState
{
    Training,
    Testing
}

public interface ITrainingTracker
{
    int CurrentEpoch { get; }
    int CurrentProgress { get; }
    EpochStats? LastEpoch { get; }
    int MaxProgress { get; }
    IReadOnlyList<EpochStats> PreviousEpochs { get; }

    public TrainingState TrainingState { get; }

    EpochStats this[int epochIndex] { get; }

    Task DisplayAsync(CancellationToken cancellationToken);

    void SetFooter(string footer);
    void SetTrainingState(TrainingState state);
    void IncrementProgress();
    void ResetProgress();
    void SetProgress(int current);
    void SetProgress(int current, int max);
    void SubmitEpoch(EpochStats stats);
}