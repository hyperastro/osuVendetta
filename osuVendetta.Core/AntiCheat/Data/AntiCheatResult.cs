namespace osuVendetta.Core.AntiCheat.Data;

public record class AntiCheatResult(AntiCheatResultType Type, string? Message = null)
{
    public static AntiCheatResult Normal()
        => Normal(null);

    public static AntiCheatResult Normal(string? message)
        => new AntiCheatResult(AntiCheatResultType.Normal, message);

    public static AntiCheatResult Relax()
        => Relax(null);

    public static AntiCheatResult Relax(string? message)
        => new AntiCheatResult(AntiCheatResultType.Relax, message);

    public static AntiCheatResult Invalid()
        => Invalid(null);

    public static AntiCheatResult Invalid(string? error)
        => new AntiCheatResult(AntiCheatResultType.Invalid, error);
}
