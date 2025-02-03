using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.Training.Utility;
public static class F1Score
{

    /// <summary>
    /// Calculates F1 score :)
    /// </summary>
    /// <param name="truePositives">The number of true positives</param>
    /// <param name="falsePositives">The number of false positives</param>
    /// <param name="falseNegatives">The number of false negatives</param>
    /// <returns>The F1 score as a float value</returns>
    public static float CalculateF1Score(int truePositives, int falsePositives, int falseNegatives)
    {
        if (truePositives == 0) return 0.0f;  // Avoid division by zero, precision is undefined in this case.

        float precision = truePositives / (float)(truePositives + falsePositives);
        float recall = truePositives / (float)(truePositives + falseNegatives);

        return 2 * (precision * recall) / (precision + recall);
    }

    /// <summary>
    /// Calculates Percision
    /// </summary>
    /// <param name="truePositives">The number of true positives</param>
    /// <param name="falsePositives">The number of false positives</param>
    /// <returns>Precision as a float value</returns>
    public static float CalculatePrecision(int truePositives, int falsePositives)
    {
        if (truePositives == 0) return 0.0f;  // Avoid division by zero, precision is undefined in this case.

        float precision = truePositives / (float)(truePositives + falsePositives);

        return precision;
    }

    /// <summary>
    /// Calculates Recall
    /// </summary>
    /// <param name="truePositives">The number of true positives</param>
    /// <param name="falseNegatives">The number of false negatives</param>
    /// <returns>Recall as a float value</returns>
    public static float CalculateRecall(int truePositives, int falseNegatives)
    {
        if (truePositives == 0) return 0.0f;  // Avoid division by zero, precision is undefined in this case.

        float recall = truePositives / (float)(truePositives + falseNegatives);

        return recall;
    }


}

