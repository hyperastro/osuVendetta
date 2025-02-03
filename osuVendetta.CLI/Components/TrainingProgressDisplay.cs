using osuVendetta.Core.Training;
using Spectre.Console;
using Spectre.Console.Rendering;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.CLI.Components;

public class TrainingProgressDisplay : ITrainingTracker
{
    public EpochStats? LastEpoch { get; private set; }
    public int CurrentEpoch => _epochStats.Count;

    public int CurrentProgress { get; private set; }
    public int MaxProgress { get; private set; }
    public TrainingState TrainingState { get; private set; }
    public IReadOnlyList<EpochStats> PreviousEpochs => _epochStats;

    readonly List<EpochStats> _epochStats;

    Table? _contentProgress;
    Table? _contentStats;
    Table? _contentConfusionMatrix;
    Table? _contentMain;
    Table? _contentStatsForEpoch;
    Table? _contentFooter;

    readonly Layout? _rootLayout;

    public TrainingProgressDisplay()
    {
        _epochStats = new List<EpochStats>();
        _rootLayout = CreateLayout();
    }

    public EpochStats this[int epochIndex]
    {
        get => _epochStats[epochIndex];
    }

    public void SetFooter(string footer)
    {
        _contentFooter?.UpdateCell(0, 0, footer);
    }

    public void SetTrainingState(TrainingState state)
    {
        if (state == TrainingState)
            return;

        TrainingState = state;
        _contentMain?.UpdateCell(0, 0, $"Training Status: {state}");
    }

    public void IncrementProgress()
    {
        SetProgress(CurrentProgress + 1);
    }

    public void ResetProgress()
    {
        SetProgress(0);
    }

    public void SetProgress(int current)
    {
        SetProgress(current, MaxProgress);
    }

    public void SetProgress(int current, int max)
    {
        CurrentProgress = current;
        MaxProgress = max;

        if (_contentProgress is null)
            return;

        int totalProgressLines = 50;
        double progressPerLine = 100.0 / totalProgressLines;
        double progress = current / (double)max * 100;
        int positiveLines = (int)(progress / progressPerLine);

        StringBuilder sb = new StringBuilder();

        sb.Append(">");
        sb.Append('X', totalProgressLines);

        if (positiveLines > 0)
        {
            sb.Insert(positiveLines + 1, "[/]");
            sb.Insert(1, "[green]");
        }

        sb.Append("<[/]");
        sb.Insert(0, "[bold]");

        _contentProgress.UpdateCell(0, 0, $"Epoch: {CurrentEpoch}");
        _contentProgress.UpdateCell(1, 0, $"Replay: {current}/{max}");
        _contentProgress.UpdateCell(3, 0, $"Progress: {progress:n2}% {sb}");
    }

    public void SubmitEpoch(EpochStats stats)
    {
        LastEpoch = stats;
        _epochStats.Add(stats);

        _contentStatsForEpoch?.UpdateCell(0, 0, $"Stats for epoch: {CurrentEpoch - 1}");

        _contentStats?.UpdateCell(0, 1, $"{stats.F1Score}");
        _contentStats?.UpdateCell(1, 1, $"{(stats.AverageProbability * 100)}");

        _contentConfusionMatrix?.UpdateCell(0, 1, $"[green]{stats.TrueNegatives}[/]");
        _contentConfusionMatrix?.UpdateCell(0, 2, $"[red]{stats.FalsePositives}[/]");
        _contentConfusionMatrix?.UpdateCell(1, 1, $"[red]{stats.FalseNegatives}[/]");
        _contentConfusionMatrix?.UpdateCell(1, 2, $"[green]{stats.TruePositives}[/]");
    }

    public async Task DisplayAsync(CancellationToken cancellationToken)
    {
        if (_rootLayout is null)
            return;

        LiveDisplay display = AnsiConsole.Live(_rootLayout);
        await display.StartAsync(async d =>
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                d.Refresh();
                await Task.Delay(15);
            }
        });
    }

    Layout CreateLayout()
    {
        _contentStats = new Table()
            .HideHeaders()
            .NoBorder()
            .AddColumns(string.Empty, string.Empty)
            .AddRow("F1 Score", string.Empty)
            .AddRow("Average Probability", string.Empty);

        _contentConfusionMatrix = new Table()
            .Title("Confusion Matrix")
            .ShowRowSeparators()
            .AddColumns(string.Empty, "Negative", "Positive")
            .AddRow("Negative",
                    string.Empty,
                    string.Empty)
            .AddRow("Positive",
                    string.Empty,
                    string.Empty);

        _contentProgress = new Table()
           .HideHeaders()
           .NoBorder()
           .AddColumn(string.Empty)
           .AddRow("Epoch:")
           .AddRow($"Replay:")
           .AddEmptyRow()
           .AddRow($"|--------------------| 0%")
           .AddEmptyRow();

        _contentStatsForEpoch = new Table()
            .HideHeaders()
            .NoBorder()
            .AddColumn(string.Empty)
            .AddRow($"Stats for epoch: {CurrentEpoch}");

        _contentFooter = new Table()
            .HideHeaders()
            .NoBorder()
            .AddColumn(string.Empty)
            .AddEmptyRow();

        _contentMain = new Table()
            .HideHeaders()
            .NoBorder()
            .AddColumn(string.Empty)
            .AddRow("Training model...")
            .AddEmptyRow()
            .AddRow(_contentProgress)
            .AddEmptyRow()
            .AddRow(new Rule())
            .AddEmptyRow()
            .AddRow(_contentStatsForEpoch)
            .AddEmptyRow()
            .AddRow(_contentConfusionMatrix)
            .AddEmptyRow()
            .AddRow(_contentStats)
            .AddEmptyRow()
            .AddRow(_contentFooter);

        return new Layout().Update(_contentMain);
    }
}
