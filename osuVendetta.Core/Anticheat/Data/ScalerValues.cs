namespace osuVendetta.Core.Anticheat.Data;

/// <summary>
/// Scaler for anticheat values
/// </summary>
public class ScalerValues
{
    public required float DeltaTime { get; set; }
    public required float X { get; set; }
    public required float Y { get; set; }
    public required float DeltaX { get; set; }
    public required float DeltaY { get; set; }
}