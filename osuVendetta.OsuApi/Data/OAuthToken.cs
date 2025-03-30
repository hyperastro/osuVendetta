using System.Text.Json.Serialization;

namespace osuVendetta.OsuApi.Data;

internal class OAuthToken
{
    [JsonPropertyName("token_type")]
    public required string TokenType { get; set; }

    [JsonPropertyName("expires_in")]
    public required long ExpiresIn { get; set; }

    [JsonPropertyName("access_token")]
    public required string AccessToken { get; set; }
}