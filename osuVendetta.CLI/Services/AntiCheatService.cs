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
        //AnsiConsole.WriteLine("Loading anticheat model...");

        //using FileStream modelStream = File.OpenRead("Data/128x3V2.safetensors");
        //_antiCheatModel.Load(modelStream);

        //AnsiConsole.WriteLine("Anticheat model loaded");
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
