using System.Reflection;
using System.Text.Json;
using osuVendetta.Core.Replays.Data;

namespace osuVendetta.CoreLib.AntiCheat.Data;

/// <summary>
/// Version of the model and config
/// </summary>
public class ModelVersion
{
    /// <summary>
    /// Text displayed for the user
    /// </summary>
    public required string DisplayText { get; set; }
    /// <summary>
    /// Any changes that are not related to the model structure should increase this value
    /// <para>When this value increases, set <see cref="Minor"/> to 0</para>
    /// </summary>
    public required int Major {  get; set; }
    /// <summary>
    /// Any changes that are not related to the model structure should increase this value
    /// </summary>
    public required int Minor {  get; set; }

}
/// <summary>
/// Config required for the model to correctly run the anticheat
/// </summary>
public class AntiCheatConfig
{
    /// <summary>
    /// <inheritdoc cref="ModelVersion"/>
    /// </summary>
    public required ModelVersion Version { get; set; }

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
    public required Scaler ScalerMean { get; set; }
    /// <summary>
    /// Value scaler for std
    /// </summary>
    public required Scaler ScalerStd { get; set; }

    /// <summary>
    /// Model batch count
    /// </summary>
    public required int BatchCount { get; set; }

    static readonly JsonSerializerOptions _jsonOptions = new JsonSerializerOptions
    {
        WriteIndented = true
    };

    public static AntiCheatConfig? FromJson(string json)
    {
        return JsonSerializer.Deserialize<AntiCheatConfig>(json, _jsonOptions);
    }

    public string ToJson()
    {
        return JsonSerializer.Serialize(this, _jsonOptions);
    }
}