namespace osuVendetta.Core.AntiCheat.Data;

public class BaseAntiCheatResult
{

}

public class AntiCheatUnhandledError : BaseAntiCheatResult
{
    public required string Error { get; set; }
}

public class AntiCheatNotSupportedRulesetError : BaseAntiCheatResult
{
    public required string RequestedRuleset { get; set; }
    public required string SupportedRulesets { get; set; }
}

public class AntiCheatResult : BaseAntiCheatResult
{
    public required AntiCheatProbability CheatProbability { get; set; }
}