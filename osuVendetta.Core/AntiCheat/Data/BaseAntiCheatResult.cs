namespace osuVendetta.Core.AntiCheat.Data;

public class BaseAntiCheatResult
{
    public override string ToString()
    {
        return "Unkown Result";
    }
}

public class AntiCheatUnhandledError : BaseAntiCheatResult
{
    public required string Error { get; set; }

    public override string ToString()
    {
        return $"Unhandled Error: {Error}";
    }
}

public class AntiCheatNotSupportedRulesetError : BaseAntiCheatResult
{
    public required string RequestedRuleset { get; set; }
    public required string SupportedRulesets { get; set; }

    public override string ToString()
    {
        return $"Unsupported ruleset: {RequestedRuleset}, Supported rulesets: {SupportedRulesets}";
    }
}

public class AntiCheatResult : BaseAntiCheatResult
{
    public required AntiCheatProbability CheatProbability { get; set; }

    public override string ToString()
    {
        return $"Normal Probability: {CheatProbability.Normal}, Relax Probability {CheatProbability.Relax}";
    }
}