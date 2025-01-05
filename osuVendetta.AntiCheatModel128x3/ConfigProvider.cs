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
                Major = 1,
                Minor = 1
            },

            InputSize = 6,
            HiddenSize = 128,
            OutputSize = 1,
            LayerCount = 3,
            Dropout = 0.3,

            StepOverlay = 500,
            StepsPerChunk = 1000,
            FeaturesPerStep = 6,

            ScalerMean = new ScalerValues
            {
                DimensionDeltaTime = 14.091803334053385f,
                DimensionX = 255.94747118245044f,
                DimensionY = 192.43203651627053f,
                DimensionDeltaX = 0.006618787601832843f,
                DimensionDeltaY = 0.06667060675316294f,
            },
            ScalerStd = new ScalerValues
            {
                DimensionDeltaTime = 2178.834268435404f,
                DimensionX = 211.089933155007f,
                DimensionY = 363.27989919306765f,
                DimensionDeltaX = 43.42019885061354f,
                DimensionDeltaY = 156.50799405495763f,
            }
        };
    }
}
