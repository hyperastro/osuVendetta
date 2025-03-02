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
                DeltaTime = 26.268179f,
                X = 26.91232f,
                Y = 26.112173f,
                DeltaX = 26.447939f,
                DeltaY = 26.407267f,
            },
            StandardDeviation = new ScalerValues
            {
                DeltaTime = 166.9607f,
                X = 341.29846f,
                Y = 333.1856f,
                DeltaX = 286.3734f,
                DeltaY = 142.87782f,
            }
        };
    }
}
