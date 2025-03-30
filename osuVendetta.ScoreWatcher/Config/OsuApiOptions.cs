using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.ScoreWatcher.Config;

internal class OsuApiOptions
{
    public static readonly string OsuApi = "OsuApi";

    /// <summary>
    /// Osu api v2 client id
    /// </summary>
    public required int ClientId { get; set; }
    /// <summary>
    /// Osu api v2 client token
    /// </summary>
    public required string ClientToken { get; set; }
}
