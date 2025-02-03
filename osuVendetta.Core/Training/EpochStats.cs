namespace osuVendetta.Core.Training;

/// <summary>
/// 
/// </summary>
/// <param name="F1Score">F1 score</param>
/// <param name="AverageProbability">Average probability</param>
public record class EpochStats(
        int Epoch,
        float F1Score,
        int TruePositives,
        int TrueNegatives,
        int FalsePositives,
        int FalseNegatives,
        float AverageProbability);
