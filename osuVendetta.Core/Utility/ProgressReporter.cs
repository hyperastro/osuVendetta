namespace osuVendetta.Core.Utility;

public class ProgressReporter : IProgressReporter
{
    public Action<double> IncrementAction { get; init; }
    public Action<double> SetCurrentProgressAction { get; init; }
    public Action<double> SetMaxProgressAction { get; init; }
    public Action<string> SetProgressTitleAction { get; init; }

    public ProgressReporter(Action<double> incrementAction,
                            Action<double> setCurrentProgressAction,
                            Action<double> setMaxProgressAction,
                            Action<string> setProgressTitleAction)
    {
        IncrementAction = incrementAction;
        SetCurrentProgressAction = setCurrentProgressAction;
        SetMaxProgressAction = setMaxProgressAction;
        SetProgressTitleAction = setProgressTitleAction;
    }

    public void Increment(double amount = 1)
    {
        IncrementAction(amount);
    }

    public void SetCurrentProgress(double current)
    {
        SetCurrentProgressAction(current);
    }

    public void SetMaxProgress(double max)
    {
        SetMaxProgressAction(max);
    }

    public void SetProgressTitle(string title)
    {
        SetProgressTitleAction(title);
    }
}
