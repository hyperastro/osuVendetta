using Microsoft.Extensions.Configuration;
using osuVendetta.ScoreWatcher.Config;
using osuVendetta.ScoreWatcher.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace osuVendetta.ScoreWatcher.Services;
public interface IDiscordWebhookService
{
    Task PostAsync(string user, string content, string? footer = null);
    Task PostReplayReport(WebhookOsuReport report);
}

public class DiscordWebhookService : IDiscordWebhookService
{
    readonly DiscordWebhookOptions _webhookOptions;
    readonly HttpClient _httpClient;

    public DiscordWebhookService(HttpClient httpClient, IConfiguration config)
    {
        _httpClient = httpClient;

        _webhookOptions = config.GetSection(DiscordWebhookOptions.DiscordWebhook)
                                .Get<DiscordWebhookOptions>()
                                ?? throw new ArgumentNullException(nameof(DiscordWebhookOptions));

        if (string.IsNullOrEmpty(_webhookOptions.WebhookUri))
            throw new ArgumentNullException("Webhook cannot be null or empty");
    }

    public async Task PostReplayReport(WebhookOsuReport report)
    {
        var payload = new
        {
            username = "ScoreWatcher",
            content = $"Player Id: {report.PlayerId}\n" +
                      $"Date: {report.DateTime}\n" +
                      $"Length: {report.Length:hh\\:mm\\:ss\\.ffff}\n" +
                      $"Confidence: {report.AverageProbability}\n" +
                      $"\n" +
                      $"Segment Probabilities:\n" +
                      string.Join(", ", report.Probabilities) +
                      $"\n\n" +
                      $"Score: https://osu.ppy.sh/scores/{report.ScoreId}\n" +
                      $"Replay: https://osu.ppy.sh/scores/{report.ScoreId}/download"
        };

        await PostAsync(payload);
    }

    public async Task PostAsync(string user, string content, string? footer = null)
    {
        var payload = new
        {
            username = user,
            content = content,
            footer = footer,
        };

        await PostAsync(payload);
    }

    async Task PostAsync(object message)
    {
        string json = JsonSerializer.Serialize(message);
        await PostAsync(json);
    }

    async Task PostAsync(string message)
    {
        HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, _webhookOptions.WebhookUri);
        request.Content = new StringContent(message, Encoding.UTF8, "application/json");

        await _httpClient.SendAsync(request);
    }
}
