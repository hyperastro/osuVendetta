using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.IO.Dataset;

public interface IReplayDatasetProvider : IEnumerable
{
    int TotalReplays { get; }
}
