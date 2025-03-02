namespace osuVendetta.OsuApi.Data;

public class OsuScores
{
    public Score[] scores { get; set; }
    public Cursor cursor { get; set; }
    public string cursor_string { get; set; }
}
