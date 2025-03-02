namespace osuVendetta.OsuApi.Data;

public class Score
{
    public int classic_total_score { get; set; }
    public bool preserve { get; set; }
    public bool processed { get; set; }
    public bool ranked { get; set; }
    public Maximum_Statistics maximum_statistics { get; set; }
    public Mod[] mods { get; set; }
    public Statistics statistics { get; set; }
    public int total_score_without_mods { get; set; }
    public int beatmap_id { get; set; }
    public object best_id { get; set; }
    public long id { get; set; }
    public string rank { get; set; }
    public string type { get; set; }
    public int user_id { get; set; }
    public float accuracy { get; set; }
    public int? build_id { get; set; }
    public DateTime ended_at { get; set; }
    public bool has_replay { get; set; }
    public bool is_perfect_combo { get; set; }
    public bool legacy_perfect { get; set; }
    public long? legacy_score_id { get; set; }
    public int legacy_total_score { get; set; }
    public int max_combo { get; set; }
    public bool passed { get; set; }
    public float? pp { get; set; }
    public int ruleset_id { get; set; }
    public DateTime? started_at { get; set; }
    public int total_score { get; set; }
    public bool replay { get; set; }
    public Current_User_Attributes current_user_attributes { get; set; }
}
