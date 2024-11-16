namespace osuVendetta.Core.Replays.Data;

/// <summary>
/// Scaler for anticheat values
/// </summary>
public record struct ScalerValues(
    double DimensionDeltaTime,
    double DimensionX,
    double DimensionY,
    double DimensionDeltaX,
    double DimensionDeltaY
);
