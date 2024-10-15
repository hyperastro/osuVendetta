using OsuParsers.Decoders;
using OsuParsers.Replays;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.AntiCheat.Data;
using osuVendetta.CoreLib.AntiCheat.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace osuVendetta.Wasm;

public partial class Core
{
    public static string? ModelPath => _modelPath;

    static AntiCheatConfig? _config;
    static string? _modelPath;
    static AntiCheatRunner? _antiCheatRunner;

    static async Task Main(string[] args)
    {
        Console.WriteLine("osuVendetta.Wasm loaded");
    }

    [JSImport("loadModel")]
    internal static partial Task LoadModel(string modelPath);

    [JSImport("loadConfig")]
    internal static partial Task<string> LoadConfigJson();

    [JSImport("runModel")]
    internal static partial Task<string> RunModel(double[] inputValues, int[] shape);

    [JSExport]
    internal static async Task InitializeAsync(string config, string modelPath)
    {
        _config = AntiCheatConfig.FromJson(config);
        _modelPath = modelPath;

        ArgumentNullException.ThrowIfNull(_config);

        IAntiCheatModel model = new AntiCheatModel192x2();
        _antiCheatRunner = new AntiCheatRunner(model, _config);

        await model.LoadAsync(new AntiCheatModelLoadArgs
        { 
            AntiCheatConfig = _config,
            ModelBytes = Array.Empty<byte>()
        });
    }

    [JSExport]
    static async Task<string> ProcessReplayAsync(byte[] replayData, bool runInParallel)
    {
        ArgumentNullException.ThrowIfNull(_antiCheatRunner);

        using MemoryStream replayStream = new MemoryStream(replayData);
        Replay replay = ReplayDecoder.Decode(replayStream);

        BaseAntiCheatResult result = await _antiCheatRunner.ProcessReplayAsync(replay, runInParallel);

        return JsonSerializer.Serialize(result, new JsonSerializerOptions(JsonSerializerDefaults.Web));
    }
}