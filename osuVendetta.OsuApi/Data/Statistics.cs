namespace osuVendetta.OsuApi.Data;

public class Statistics
{
    public int ok { get; set; }
    public int meh { get; set; }
    public int great { get; set; }
    public int miss { get; set; }
    public int ignore_hit { get; set; }
    public int ignore_miss { get; set; }
    public int large_bonus { get; set; }
    public int small_bonus { get; set; }
    public int large_tick_hit { get; set; }
    public int large_tick_miss { get; set; }
    public int slider_tail_hit { get; set; }
    public int small_tick_hit { get; set; }
    public int small_tick_miss { get; set; }
}
