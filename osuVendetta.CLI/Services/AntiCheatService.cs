using Microsoft.Extensions.Hosting;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace osuVendetta.CLI.Services;

/// <summary>
/// Service responsible for handling the anticheat
/// </summary>
internal class AntiCheatService : IHostedService
{
    readonly IAntiCheatModel _antiCheatModel;
    readonly IReplayProcessor _replayProcessor;

    public AntiCheatService(IAntiCheatModel antiCheatModel, IReplayProcessor replayProcessor)
    {
        _antiCheatModel = antiCheatModel;
        _replayProcessor = replayProcessor;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        AnsiConsole.WriteLine("Loading anticheat model...");

        //using FileStream modelStream = File.OpenRead("Data/128x3V2.safetensors");
        //_antiCheatModel.Load(modelStream);

        AnsiConsole.WriteLine("Anticheat model loaded");

        DeviceType device = DeviceType.CPU;
        string modelVersion = _antiCheatModel.Config.Version?.ToString() ?? "Unkown Version";

#if RELEASE_WIN_CPU

#endif

#if RELEASE_WIN_CPU || DEBUG_WIN_CPU || RELEASE_LINUX_CPU || DEBUG_LINUX_CPU
        device = DeviceType.CPU;
#elif RELEASE_WIN_CUDA || DEBUG_WIN_CUDA || RELEASE_LINUX_CUDA || DEBUG_LINUX_CUDA
        device = DeviceType.CUDA;
#endif

        AnsiConsole.WriteLine($"Model device type: {device}");
        Console.Title = $"osu!Vendetta | Device: {device}";
        _antiCheatModel.SetDevice(device);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {

    }

    public async Task<AntiCheatModelResult> RunInferenceAsync(Stream replay)
    {
        return await Task.Run(() =>
        {
            ReplayTokens tokens = _replayProcessor.CreateTokensParallel(replay);
            return _antiCheatModel.RunInference(tokens);
        });
    }
}
