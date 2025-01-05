namespace osuVendetta.Core.Utility;

public interface IProgressReporter
{
    void Increment(double amount = 1.0);
    void SetMaxProgress(double max);
    void SetCurrentProgress(double current);
    void SetProgressTitle(string title);
}
