namespace osuVendetta.Core.AntiCheat.Data;

public record class AntiCheatResultMetadata(string Player);

public class AntiCheatResult
{
    public AntiCheatResultType Type { get; init; }
    public string? Message { get; init; }
    public AntiCheatResultMetadata? Metadata { get; set; }

    public AntiCheatResult(AntiCheatResultType type) : this(type, null)
    {

    }

    public AntiCheatResult(AntiCheatResultType type, string? message) : this(type, message, null)
    {
    
    }

    public AntiCheatResult(AntiCheatResultType type, string? message, AntiCheatResultMetadata? metadata)
    {
        Type = type;
        Message = message;
        Metadata = metadata;
    }

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
