namespace osuVendetta.ScoreWatcher.Config;

internal class ScoreWatcherOptions
{
    public static readonly string ScoreWatcher = "ScoreWatcher";

    /// <summary>
    /// How long should we wait before fetching new scores if we previously didn't receive any new ones
    /// </summary>
    public int DelayIfNoScoresMs { get; set; }
    /// <summary>
    /// How many scores (with replays) we should cache from each scores fetch
    /// </summary>
    public int ScoresToCache { get; set; }
    /// <summary>
    /// How long to wait between downloads
    /// </summary>
    public int DelayBetweenDownloadsMs { get; set; }
}
