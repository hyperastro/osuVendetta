using osuVendetta.Core.Anticheat.Data;

namespace osuVendetta.AntiCheatModel128x3;

internal static class ConfigProvider
{
    public static AntiCheatModelConfig CreateConfig()
    {
        return new AntiCheatModelConfig
        {
            Version = new AntiCheatModelVersion
            {
                DisplayText = "128x3 Model",
                Major = 2,
                Minor = 0
            },

            InputSize = 6,
            HiddenSize = 128,
            OutputSize = 1,
            LayerCount = 3,
            Dropout = 0.3,

            StepOverlay = 0,
            StepsPerChunk = 500,
            FeaturesPerStep = 6,

            StandardMean = new ScalerValues
            {
                DeltaTime = 14.091803334053385f,
                X = 255.94747118245044f,
                Y = 192.43203651627053f,
                DeltaX = 0.006618787601832843f,
                DeltaY = 0.06667060675316294f,
            },
            StandardDeviation = new ScalerValues
            {
                DeltaTime = 2178.834268435404f,
                X = 211.089933155007f,
                Y = 363.27989919306765f,
                DeltaX = 43.42019885061354f,
                DeltaY = 156.50799405495763f,
            }
        };
    }
}
