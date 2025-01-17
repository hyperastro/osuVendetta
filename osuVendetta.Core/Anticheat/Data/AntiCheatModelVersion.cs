namespace osuVendetta.Core.Anticheat.Data;

/// <summary>
/// Version of the model and config
/// </summary>
public class AntiCheatModelVersion
{
    /// <summary>
    /// Text displayed for the user
    /// </summary>
    public required string DisplayText { get; set; }
    /// <summary>
    /// Any changes that are not related to the model structure should increase this value
    /// <para>When this value increases, set <see cref="Minor"/> to 0</para>
    /// </summary>
    public required int Major { get; set; }
    /// <summary>
    /// Any changes that are not related to the model structure should increase this value
    /// </summary>
    public required int Minor { get; set; }

    public override string ToString()
    {
        return DisplayText;
    }
}
