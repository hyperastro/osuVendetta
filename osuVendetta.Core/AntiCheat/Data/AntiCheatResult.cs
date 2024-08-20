namespace osuvendetta.Core.AntiCheat.Data;

public record class AntiCheatResult(AntiCheatResultType Type, string? ErrorMessage = null);
