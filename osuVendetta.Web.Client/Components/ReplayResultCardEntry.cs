namespace osuVendetta.Web.Client.Components;

public class ReplayResultCardEntry
{
    public required string PlayerName { get; set; }
    public required string ReplayFileName { get; set; }
    public required DateTime ReplayDate { get; set; }
    public required double ProbabilityOfCheating { get; set; }
}