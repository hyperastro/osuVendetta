using osuVendetta.Core.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.CLI;
internal class ConfigDir
{
    public static void OverwriteConfigDir()
    {
        BaseConfig.BasePath = Environment.CurrentDirectory;
    }
}
