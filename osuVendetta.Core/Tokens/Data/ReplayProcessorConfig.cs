using osuVendetta.Core.Anticheat.Data;
using System.Reflection;
using System.Text.Json;

namespace osuVendetta.Core.Tokens.Data;
/// <summary>
/// Config required for the model to correctly run the anticheat
/// </summary>
public class ReplayProcessorConfig
{
    /// <summary>
    /// <inheritdoc cref="AntiCheatModelVersion"/>
    /// </summary>
    public required AntiCheatModelVersion Version { get; set; }

    // 1000a
    /// <summary>
    /// Amount of steps per chunk
    /// </summary>
    public required int StepsPerChunk { get; set; }
    // 500
    /// <summary>
    /// Overlay of last chunk into current chunk
    /// </summary>
    public required int StepOverlay { get; set; }
    // 6
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

    /// <summary>
    /// Model batch count
    /// </summary>
    public required int BatchCount { get; set; }

    static readonly JsonSerializerOptions _jsonOptions = new JsonSerializerOptions
    {
        WriteIndented = true
    };

    public static ReplayProcessorConfig? FromJson(string json)
    {
        return JsonSerializer.Deserialize<ReplayProcessorConfig>(json, _jsonOptions);
    }

    public string ToJson()
    {
        return JsonSerializer.Serialize(this, _jsonOptions);
    }
}