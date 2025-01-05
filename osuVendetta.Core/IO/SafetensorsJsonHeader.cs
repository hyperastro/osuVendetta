using System.Text.Json;

namespace osuVendetta.Core.IO;

public class SafetensorsJsonHeader : Dictionary<string, SafetensorsJsonHeaderEntry>
{
    public static SafetensorsJsonHeader? FromJson(string json)
    {
        return JsonSerializer.Deserialize<SafetensorsJsonHeader>(json);
    }
}

