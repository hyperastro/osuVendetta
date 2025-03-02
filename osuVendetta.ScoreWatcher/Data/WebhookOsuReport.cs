using osuVendetta.Core.IO.Dataset;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.ScoreWatcher.Data;

public class WebhookOsuReport
{
    public required ReplayDatasetClass Class { get; set; }
    public required string PlayerId { get; set; }
    public required long ScoreId { get; set; }
    public required DateTime DateTime { get; set; }
    public required TimeSpan Length { get; set; }
    public required float AverageProbability { get; set; }
    public required float[] Probabilities { get; set; }
}
