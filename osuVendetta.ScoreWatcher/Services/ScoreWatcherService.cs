using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OsuParsers.Beatmaps.Sections;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using osuVendetta.OsuApi;
using osuVendetta.OsuApi.Data;
using osuVendetta.ScoreWatcher.Config;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;

namespace osuVendetta.ScoreWatcher.Services;

internal class ScoreWatcherService : IHostedService
{
    readonly IHostApplicationLifetime _lifetime;
    readonly IOsuApiClient _osuApi;
    readonly ILogger _logger;
    readonly IAntiCheatModel _antiCheatModel;
    readonly IReplayProcessor _replayProcessor;
    readonly IDiscordWebhookService _discordWebhookService;

    readonly ScoreWatcherOptions _scoreWatcherOptions;
    readonly OsuApiOptions _osuApiOptions;

    string _lastApiScoreCursor;

    public ScoreWatcherService(IHostApplicationLifetime lifetime, IOsuApiClient osuApi, IConfiguration config,
        IAntiCheatModel antiCheatModel, IReplayProcessor replayProcessor, IDiscordWebhookService discordWebhookService,
        ILogger<ScoreWatcherService> logger)
    {
        _lifetime = lifetime;
        _osuApi = osuApi;
        _logger = logger;
        _antiCheatModel = antiCheatModel;
        _replayProcessor = replayProcessor;
        _lastApiScoreCursor = string.Empty;
        _discordWebhookService = discordWebhookService;

        _scoreWatcherOptions = config.GetSection(ScoreWatcherOptions.ScoreWatcher)
                                     .Get<ScoreWatcherOptions>()
                                     ?? throw new ArgumentNullException(nameof(ScoreWatcherOptions));

        _osuApiOptions = config.GetSection(OsuApiOptions.OsuApi)
                               .Get<OsuApiOptions>()
                               ?? throw new ArgumentNullException(nameof(OsuApiOptions));

        ValidateSettings();
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _lifetime.ApplicationStarted.Register(async () => await RunAsync(cancellationToken));

        using FileStream modelStream = File.OpenRead("Data/model.41.bin");
        using BinaryReader modelReader = new BinaryReader(modelStream);
        _antiCheatModel.Load(modelReader);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Exiting...");
    }

    async Task RunAsync(CancellationToken cancellationToken)
    {
        if (!await HandleAuth(_osuApiOptions.ClientToken, _osuApiOptions.ClientId, cancellationToken))
        {
            _logger.LogError("Failed to authenticate");
            _lifetime.StopApplication();
            return;
        }

        _logger.LogInformation("Starting score watch");

        while (!cancellationToken.IsCancellationRequested)
        {
            List<Score> scoresToProcess = await GetScoresToProcess(cancellationToken);

            if (scoresToProcess.Count == 0)
            {
                _logger.LogDebug($"No scores found, pausing {_scoreWatcherOptions.DelayIfNoScoresMs} ms");
                await Task.Delay(_scoreWatcherOptions.DelayIfNoScoresMs);
            }

            await ProcessScores(scoresToProcess, cancellationToken);
        }
    }

    async Task<List<Score>> GetScoresToProcess(CancellationToken cancellationToken)
    {
        List<Score> scoresToProcess = new List<Score>(_scoreWatcherOptions.ScoresToCache);
        OsuScores? scores = await _osuApi.GetScoresAsync(_lastApiScoreCursor, cancellationToken);

        if (scores is null)
            return scoresToProcess;

        _logger.LogDebug($"{scores.scores.Length} scores found, getting {_scoreWatcherOptions.ScoresToCache} random replays...");

        _lastApiScoreCursor = scores.cursor_string;

        for (int i = 0; i < scores.scores.Length && i < _scoreWatcherOptions.ScoresToCache; i++)
            if (scores.scores[i].has_replay)
                scoresToProcess.Add(scores.scores[i]);

        _logger.LogInformation($"Found {scoresToProcess.Count} (Max: {_scoreWatcherOptions.ScoresToCache}) scores with replays");

        return scoresToProcess;
    }

    async Task ProcessScores(List<Score> scores, CancellationToken cancellationToken)
    {
        for (int i = 0; i < scores.Count; i++)
        {
            Score score = scores[i];
            using Stream? replayDownloadStream = await _osuApi.GetReplayAsync((ulong)score.id, cancellationToken);

            if (replayDownloadStream is null)
            {
                _logger.LogWarning($"Failed to get replay for score {score.id} ({(ulong)score.id})");
                continue;
            }

            using MemoryStream replayStream = new MemoryStream();
            await replayDownloadStream.CopyToAsync(replayStream);
            replayStream.Seek(0, SeekOrigin.Begin);

            try
            {
                _logger.LogInformation($"Processing score {score.id}");

                ReplayTokens tokens = _replayProcessor.CreateTokensParallel(replayStream);
                AntiCheatModelResult inferenceResult = _antiCheatModel.RunInference(tokens);

                await ProcessScoreResult(score, inferenceResult);

                _logger.LogInformation($"Waiting {_scoreWatcherOptions.DelayBetweenDownloadsMs} before next download");
                await Task.Delay(_scoreWatcherOptions.DelayBetweenDownloadsMs);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to process score {scores[i].id}: {ex}");

                byte[] data = replayStream.ToArray();

                if (data.Length > 0)
                {
                    string textForm = Encoding.UTF8.GetString(data);

                    if (!string.IsNullOrEmpty(textForm))
                    _logger.LogError($"Extra info: {textForm}");
                }
            }
        }
    }

    async Task ProcessScoreResult(Score score, AntiCheatModelResult result)
    {
        using torch.Tensor resultSigmoid = torch.sigmoid(result.Segments);
        float[] segments = resultSigmoid.ToArray<float>();

        float averageProbability = 0;

        for (int i = 0; i < segments.Length; i++)
        {
            averageProbability += segments[i];
        }

        averageProbability /= segments.Length;

        _logger.LogInformation($"Score {score.id} has an average probability of {averageProbability} %");

        if (averageProbability < 0.55)
            return;

        await ReportScore(score, averageProbability, segments);
    }

    async Task ReportScore(Score score, float averageProbability, float[] probabilities)
    {
        _logger.LogInformation($"Reporting score {score.id}");

        await _discordWebhookService.PostReplayReport(new Data.WebhookOsuReport
        {
            ScoreId = score.id,
            Class = Core.IO.Dataset.ReplayDatasetClass.Relax,
            DateTime = score.ended_at,
            Length = score.ended_at - (score.started_at ?? score.ended_at),
            PlayerId = score.user_id.ToString(),
            
            AverageProbability = averageProbability,
            Probabilities = probabilities
        });
    }

    async Task<bool> HandleAuth(string clientSecret, int clientId, CancellationToken cancellationToken)
    {
        bool authStatus = await _osuApi.AuthenticateAnonymousAsync(clientId, clientSecret, cancellationToken);

        if (!authStatus)
        {
            _logger.LogError("Failed to authenticate");
            _lifetime.StopApplication();

            return false;
        }

        return true;
    }


    void ValidateSettings()
    {
        if (string.IsNullOrEmpty(_osuApiOptions.ClientToken))
            throw new ArgumentNullException(_osuApiOptions.ClientToken, "Cannot be null or empty");

        if (_osuApiOptions.ClientId <= 0)
            throw new ArgumentOutOfRangeException(nameof(_osuApiOptions.ClientId), "Cannot be <= 0");

        if (_scoreWatcherOptions.ScoresToCache <= 0)
            throw new ArgumentOutOfRangeException(nameof(_scoreWatcherOptions.ScoresToCache), "Cannot be <= 0");

        if (_scoreWatcherOptions.DelayIfNoScoresMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(_scoreWatcherOptions.DelayIfNoScoresMs), "Cannot be <= 0");

        if (_scoreWatcherOptions.DelayBetweenDownloadsMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(_scoreWatcherOptions.DelayBetweenDownloadsMs), "Cannot be <= 0");
    }

}
