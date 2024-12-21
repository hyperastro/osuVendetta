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
                Minor = 0
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
                DimensionDeltaTime = 7.03284752f,
                DimensionX = 2.55463221e+02f,
                DimensionY = 1.92135235e+02f,
                DimensionDeltaX = 0.00209958789f,
                DimensionDeltaY = 0.0268233697f,
            },
            ScalerStd = new ScalerValues
            {
                DimensionDeltaTime = 562.68982467f,
                DimensionX = 188.44789912f,
                DimensionY = 173.43673961f,
                DimensionDeltaX = 27.54802019f,
                DimensionDeltaY = 27.51032391f,
            }
        };
    }
}
