using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using osuVendetta.Core.AntiCheat.Data;
using osuVendetta.CoreLib.AntiCheat.Data;

namespace osuVendetta.Core.AntiCheat;

public class AntiCheatModelLoadArgs
{
    public required byte[] ModelBytes { get; set; }
    public required AntiCheatConfig AntiCheatConfig { get; set; }
}

public interface IAntiCheatModel
{
    Task LoadAsync(AntiCheatModelLoadArgs loadArgs);
    Task UnloadAsync();
    Task<Logit> Run(ModelInput input);
}
