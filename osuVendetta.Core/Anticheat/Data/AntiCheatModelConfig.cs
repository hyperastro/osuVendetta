namespace osuVendetta.Core.Anticheat.Data;

public class AntiCheatModelConfig
{
    /// <summary>
    /// <inheritdoc cref="AntiCheatModelVersion"/>
    /// </summary>
    public required AntiCheatModelVersion Version { get; set; }

    public required int HiddenSize { get; set; }
    public required int LayerCount { get; set; }
    public required int InputSize { get; set; }
    public required int OutputSize { get; set; }
    public required double Dropout { get; set; }

    /// <summary>
    /// Amount of steps per chunk
    /// </summary>
    public required int StepsPerChunk { get; set; }
    /// <summary>
    /// Overlay of last chunk into current chunk
    /// </summary>
    public required int StepOverlay { get; set; }
    /// <summary>
    /// Features for each step
    /// </summary>
    public required int FeaturesPerStep { get; set; }
    /// <summary>
    /// Total features contained in one chunk
    /// </summary>
    public int TotalFeatureSizePerChunk => StepsPerChunk * FeaturesPerStep;
    /// <summary>
    /// Value scaler for mean
    /// </summary>
    public required ScalerValues ScalerMean { get; set; }
    /// <summary>
    /// Value scaler for std
    /// </summary>
    public required ScalerValues ScalerStd { get; set; }
}
