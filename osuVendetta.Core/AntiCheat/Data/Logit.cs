namespace osuVendetta.Core.AntiCheat.Data;

/// <summary>
/// Logit representing model output
/// </summary>
public struct Logit
{
    public required float Relax;
    public required float Normal;

    public Logit(float relax, float normal) : this()
    {
        Relax = relax;
        Normal = normal;
    }
}
