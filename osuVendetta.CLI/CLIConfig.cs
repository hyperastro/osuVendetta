using osuVendetta.Core.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.CLI;
public class CLIConfig : BaseConfig
{
    public string TrainingNormalDatasetDirectory { get; set; } = "Data/Training/V1/Normal";
    public string TrainingRelaxDatasetDirectory { get; set; } = "Data/Training/V1/Relax";
    public string ModelBenchmarkNormalDatasetDirectory { get; set; } = "Data/Benchmark/V1/Normal";
    public string ModelBenchmarkRelaxDatasetDirectory { get; set; } = "Data/Benchmark/V1/Relax";
    public string GlobalBenchmarkDir { get; set; } = "Data/Benchmark";
    public string ModelPath { get; set; } = "Data/128x3V2.safetensors";
}
