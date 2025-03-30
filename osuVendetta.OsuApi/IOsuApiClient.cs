
using osuVendetta.OsuApi.Data;

namespace osuVendetta.OsuApi;

public interface IOsuApiClient
{
    Task<bool> AuthenticateAnonymousAsync(int clientId, string clientSecret, CancellationToken cancellationToken);
    Task<OsuScores?> GetScoresAsync(CancellationToken cancellationToken);
    Task<OsuScores?> GetScoresAsync(string cursorString, CancellationToken cancellationToken);
    Task<Stream?> GetReplayAsync(ulong scoreId, CancellationToken cancellationToken);
}
