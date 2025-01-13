using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace osuVendetta.Core.IO.Safetensors;
public class Safetensors
{
    public SafetensorsJsonHeader Header { get; private set; }
    public byte[] Data { get; private set; }

    Safetensors(SafetensorsJsonHeader header, byte[] data)
    {
        Header = header;
        Data = data;
    }

    public static Safetensors Load(string file)
    {
        using FileStream fileStream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);

        return Load(fileStream);
    }

    public static Safetensors Load(Stream stream)
    {
        using BinaryReader reader = new BinaryReader(stream);

        long headerSize = reader.ReadInt64();
        string headerJson = Encoding.UTF8.GetString(reader.ReadBytes((int)headerSize));

        SafetensorsJsonHeader header = SafetensorsJsonHeader.FromJson(headerJson) ??
            throw new InvalidOperationException("Header of safetensors is null");

        byte[] data = new byte[stream.Length - stream.Position];
        stream.ReadExactly(data);

        return new Safetensors(header, data);
    }

    public static Safetensors FromStateDict(Dictionary<string, Tensor> stateDict)
    {
        // TODO: convert state dict to safetensors
        throw new NotImplementedException();
    }

    public void Save(Stream stream)
    {
        // TODO: save safetensors
        throw new NotImplementedException();
    }

    public byte[] GetDataFromHeader(SafetensorsJsonHeaderEntry entry)
    {
        int start = entry.DataOffsets[0];
        int end = entry.DataOffsets[1];
        int length = end - start;

        byte[] data = new byte[length];
        Buffer.BlockCopy(Data, start, data, 0, length);

        return data;
    }

    public Dictionary<string, Tensor> ToStateDict()
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(Header.Count);

        foreach ((string key, SafetensorsJsonHeaderEntry entry) in Header)
        {
            byte[] data = GetDataFromHeader(entry);
            float[] tensorData = new float[data.Length / 4];

            for (int i = 0; i < tensorData.Length; i++)
                tensorData[i] = BitConverter.ToSingle(data, i * 4);

            Tensor tensor = torch.tensor(tensorData, dimensions: entry.Shape.Select(v => (long)v).ToArray());

            // translate key so it fits the c# style
            if (!key.StartsWith('_'))
                result.Add($"_{key}", tensor);
            else
                result.Add(key, tensor);
        }

        return result;
    }

}
