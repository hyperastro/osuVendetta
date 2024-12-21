namespace osuVendetta.Core.Replays.Data;

public record class ReplayValidationResult(bool IsValid, string? ValidationError);