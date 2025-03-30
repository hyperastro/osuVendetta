using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.ScoreWatcher.Config;

internal class DiscordWebhookOptions
{
    public static readonly string DiscordWebhook = "DiscordWebhook";

    public required string WebhookUri { get; set; }
}
