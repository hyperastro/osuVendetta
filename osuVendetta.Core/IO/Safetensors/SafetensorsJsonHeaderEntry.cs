using System.Text;
using System.Text.Json.Serialization;

namespace osuVendetta.Core.IO.Safetensors;

public class SafetensorsJsonHeaderEntry
{
    [JsonPropertyName("dtype")]
    public required string DType { get; set; }
    [JsonPropertyName("shape")]
    public required int[] Shape { get; set; }
    [JsonPropertyName("data_offsets")]
    public required int[] DataOffsets { get; set; }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();

        sb.AppendLine($"DType: {DType}");

        sb.AppendLine("Shape: ");
        foreach (int shape in Shape)
            sb.Append($"{shape}, ");
        sb.Remove(sb.Length - 2, 2);
        sb.AppendLine();

        sb.AppendLine("DataOffsets: ");
        foreach (int dataOffset in DataOffsets)
            sb.Append($"{dataOffset}, ");
        sb.Remove(sb.Length - 2, 2);
        sb.AppendLine();

        return sb.ToString();
    }
}

