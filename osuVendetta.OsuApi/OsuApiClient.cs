using Microsoft.Extensions.Logging;
using osuVendetta.OsuApi.Data;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.Linq;
using System.Net.Mime;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace osuVendetta.OsuApi;

public class OsuApiClient : IOsuApiClient
{
    readonly static string _apiBaseUri = "https://osu.ppy.sh/api/v2/";
    readonly static string _apiOAuthUri = "https://osu.ppy.sh/oauth/";

    readonly SemaphoreSlim _rateLimit;
    readonly Timer _rateLimitTimer;
    readonly HttpClient _client;
    readonly ILogger _logger;

    OAuthToken? _authToken;

    public OsuApiClient(HttpClient httpClient, ILogger<OsuApiClient> logger)
    {
        _rateLimit = new SemaphoreSlim(60, 60);
        _rateLimitTimer = new Timer(OnRateLimitTimer, null, 0, 60 * 1000);
        _client = httpClient;
        _logger = logger;
    }

    public async Task<bool> AuthenticateAnonymousAsync(int clientId, string clientSecret, CancellationToken cancellationToken)
    {
        _logger.LogInformation("Authenticating");

        string requestContent = $"client_id={clientId}" +
                                $"&client_secret={clientSecret}" +
                                $"&grant_type=client_credentials" +
                                $"&scope=public";

        HttpRequestMessage request = CreateRequestMessage(HttpMethod.Post, _apiOAuthUri + "token", requestContent, false);
        HttpResponseMessage response = await RequestAsync(request, cancellationToken);

        if (response.StatusCode != System.Net.HttpStatusCode.OK)
        {
            _logger.LogError("Failed to authenticate");
            return false;
        }

        string content = await response.Content.ReadAsStringAsync();
        _authToken = JsonSerializer.Deserialize<OAuthToken>(content);

        _logger.LogInformation("Authentication success");
        return true;
    }

    public async Task<OsuScores?> GetScoresAsync(CancellationToken cancellationToken)
    {
        return await GetScoresAsync(string.Empty, cancellationToken);
    }

    public async Task<OsuScores?> GetScoresAsync(string cursorString, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"Requesting latest scores");

        string uriPath = $"scores?" +
                         $"ruleset=osu" +
                         $"&cursor_string={cursorString}";

        HttpRequestMessage request = CreateRequestMessage(HttpMethod.Get, _apiBaseUri + uriPath);
        HttpResponseMessage response = await RequestAsync(request, cancellationToken);
        string content = await response.Content.ReadAsStringAsync();

        return JsonSerializer.Deserialize<OsuScores>(content);
    }

    public async Task<Stream?> GetReplayAsync(ulong scoreId, CancellationToken cancellationToken)
    {
        _logger.LogInformation($"Requesting replay for score {scoreId}");

        string uriPath = $"scores/{scoreId}/download";

        HttpRequestMessage request = CreateRequestMessage(HttpMethod.Get, _apiBaseUri + uriPath);
        HttpResponseMessage response = await RequestAsync(request, cancellationToken);

        return await response.Content.ReadAsStreamAsync();
    }

    async Task<HttpResponseMessage> RequestAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        await _rateLimit.WaitAsync();
        return await _client.SendAsync(request, cancellationToken);
    }

    void OnRateLimitTimer(object? state)
    {
        while (_rateLimit.CurrentCount > 0)
            _ = _rateLimit.WaitAsync();

        _rateLimit.Release(60);
    }

    HttpRequestMessage CreateRequestMessage(HttpMethod method, string uri, string content, bool applyAuth = true)
    {
        return CreateRequestMessage(method, uri, new StringContent(content), applyAuth);
    }

    HttpRequestMessage CreateRequestMessage(HttpMethod method, string uri, HttpContent? content = null, bool applyAuth = true)
    {
        HttpRequestMessage message = new HttpRequestMessage(method, uri);

        message.Headers.Clear();
        message.Headers.Add("Accept", "application/json");

        message.Content = content;
        message.Content?.Headers.Clear();
        message.Content?.Headers.Add("Content-Type", "application/x-www-form-urlencoded");

        if (applyAuth)
        {
            if (_authToken is null)
                throw new InvalidOperationException("Cannot set auth token when auth token is null");

            message.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _authToken.AccessToken);
        }

        return message;
    }
}
