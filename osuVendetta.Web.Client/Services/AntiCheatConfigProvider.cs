using osuVendetta.CoreLib.AntiCheat.Data;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace osuVendetta.Web.Client.Services;

public interface IAntiCheatConfigProvider
{
    Task<AntiCheatConfig?> GetConfig();
    Task<string?> GetModelPath();
}

public class AntiCheatConfigProvider : IAntiCheatConfigProvider
{
    const string _CONFIG_PATH = "/antiCheat/192x2Config.json";
    const string _MODEL_PATH = "/api/File?file=192x2.onnx";

    readonly HttpClient _httpClient;

    public AntiCheatConfigProvider(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<AntiCheatConfig?> GetConfig()
    {
        string json = await _httpClient.GetStringAsync(_CONFIG_PATH);
        return JsonSerializer.Deserialize<AntiCheatConfig>(json);
    }

    public async Task<string?> GetModelPath()
    {
        return _MODEL_PATH;
    }
}