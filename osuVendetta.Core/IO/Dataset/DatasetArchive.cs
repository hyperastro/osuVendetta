using OsuParsers.Replays;
using osuVendetta.Core.Replays;
using osuVendetta.Core.Replays.Data;
using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace osuVendetta.Core.IO.Dataset;

public unsafe class DatasetArchive
{
    /*
        4 bytes header
        24 x N bytes header entries
        ? x N bytes content

        header entry
            start -> from filestart
     */

    [StructLayout(LayoutKind.Sequential)]
    record struct DatasetHeader(
        int TotalEntries);

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    record struct DatasetHeaderEntry(
        int Class,
        long Start,
        int Length);

    public int Count => _header->TotalEntries;

    readonly MemoryMappedFile _mmf;
    readonly MemoryMappedViewAccessor _accessor;
    readonly byte* _fileStart;

    readonly DatasetHeader* _header;
    readonly DatasetHeaderEntry* _headerEntries;
    readonly byte* _content;

    public FileInfo File { get; init; }

    DatasetArchive(FileInfo file) 
    {
        _mmf = MemoryMappedFile.CreateFromFile(file.FullName);
        _accessor = _mmf.CreateViewAccessor();
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _fileStart);

        byte* pos = _fileStart;

        _header = (DatasetHeader*)pos;
        pos += sizeof(DatasetHeader);

        _headerEntries = (DatasetHeaderEntry*)pos;
        _content = (byte*)(_headerEntries + _header->TotalEntries);
    }

    public ReplayDatasetEntry this[int index]
    {
        get => GetDatasetEntry(index);
    }

    public void ExportContentInfo(FileInfo destTxtFile)
    {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < Count; i++)
        {
            ReplayDatasetEntry entry = this[i];
            sb.AppendLine($"{i}: {entry.Class}");
        }

       System.IO.File.WriteAllText(destTxtFile.FullName, sb.ToString());
    }

    public static DatasetArchive Load(FileInfo file)
    {
        return new DatasetArchive(file);
    }

    public static void Create(FileInfo file, Dictionary<ReplayDatasetClass, List<FileInfo>> datasets,
        IReplayProcessor replayProcessor, IProgress<double> progress)
    {
        if (file.Exists)
            file.Delete();

        int totalReplays = datasets.Sum(d => d.Value.Count);
        long contentOffset = sizeof(DatasetHeader) + sizeof(DatasetHeaderEntry) * totalReplays;

        using FileStream archiveStream = file.Create();

        DatasetHeader header = new DatasetHeader(totalReplays);
        archiveStream.Write(new Span<byte>(&header, sizeof(DatasetHeader)));

        // set min file length
        archiveStream.SetLength(contentOffset + 1);

        long currentHeaderOffset = sizeof(DatasetHeader);
        long currentContentOffset = contentOffset;
        int processedReplays = 0;

        foreach (KeyValuePair<ReplayDatasetClass, List<FileInfo>> dataset in datasets)
        {
            foreach (FileInfo replayFile in dataset.Value)
            {
                using FileStream replayStream = replayFile.OpenRead();
                byte[] tokenData;

                // temporary, just for old dataset
                if (replayFile.Extension.Equals(".txt"))
                {
                    using StreamReader replayReader = new StreamReader(replayStream);
                    List<byte> tempTokenData = new List<byte>();

                    bool justContinue = false;

                    string? line;
                    while ((line = replayReader.ReadLine()) is not null)
                    {
                        string[] split = line.Split(',');

                        if (split.Length < 6)
                            continue;

                        List<float> tempTokens = new List<float>(5);

                        // if we have any NAN values then replace them with 0
                        AddIgnoreNan(tempTokens, split[0], 
                            replayProcessor.AntiCheatModelConfig.ScalerStd.DimensionDeltaTime,
                            replayProcessor.AntiCheatModelConfig.ScalerMean.DimensionDeltaTime);
                        AddIgnoreNan(tempTokens, split[1],
                            replayProcessor.AntiCheatModelConfig.ScalerStd.DimensionX,
                            replayProcessor.AntiCheatModelConfig.ScalerMean.DimensionX);
                        AddIgnoreNan(tempTokens, split[2],
                            replayProcessor.AntiCheatModelConfig.ScalerStd.DimensionY,
                            replayProcessor.AntiCheatModelConfig.ScalerMean.DimensionY);
                        AddIgnoreNan(tempTokens, split[3],
                            replayProcessor.AntiCheatModelConfig.ScalerStd.DimensionDeltaX,
                            replayProcessor.AntiCheatModelConfig.ScalerMean.DimensionDeltaX);
                        AddIgnoreNan(tempTokens, split[4],
                            replayProcessor.AntiCheatModelConfig.ScalerStd.DimensionDeltaY,
                            replayProcessor.AntiCheatModelConfig.ScalerMean.DimensionDeltaY);

                        switch (split[5].ToLower())
                        {
                            case "m1":
                                tempTokens.Add(0);
                                break;

                            case "m1m2":
                                tempTokens.Add(1);
                                break;

                            case "m2":
                                tempTokens.Add(2);
                                break;

                            default:
                            case "none":
                                tempTokens.Add(3);
                                break;
                        }

                        foreach (float token in tempTokens)
                            tempTokenData.AddRange(BitConverter.GetBytes(token));
                    }

                    if (justContinue)
                        continue;

                    tokenData = tempTokenData.ToArray();
                }
                else
                {
                    ReplayTokens tokens = replayProcessor.CreateTokensParallel(replayStream);
                    tokenData = new byte[tokens.Tokens.Length * sizeof(float)];

                    for (int i = 0; i < tokens.Tokens.Length; i++)
                        _ = BitConverter.TryWriteBytes(
                            new Span<byte>(tokenData, i * sizeof(float), sizeof(float)),
                            tokens.Tokens[i]);
                }

                archiveStream.Seek(currentHeaderOffset, SeekOrigin.Begin);

                DatasetHeaderEntry headerEntry = new DatasetHeaderEntry((int)dataset.Key, currentContentOffset, tokenData.Length);
                archiveStream.Write(new Span<byte>(&headerEntry, sizeof(DatasetHeaderEntry)));
                currentHeaderOffset += sizeof(DatasetHeaderEntry);

                archiveStream.Seek(currentContentOffset, SeekOrigin.Begin);
                archiveStream.Write(tokenData);

                currentContentOffset += tokenData.Length;

                processedReplays++;
                progress.Report(processedReplays);
            }
        }
    }

    static void AddIgnoreNan(List<float> values, string input, float scalerStd, float scalerMean)
    {
        float value = float.Parse(input);

        if (float.IsNaN(value))
        {
            values.Add(0);
        }
        else
        {
            value = ReplayProcessor.Normalize(value, scalerMean, scalerStd);
            values.Add(value);
        }
    }

    ReplayDatasetEntry GetDatasetEntry(int index)
    {
        DatasetHeaderEntry* header = _headerEntries + index;
        byte* pos = _fileStart + header->Start;

        float[] data = new float[header->Length / sizeof(float)];

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = BitConverter.ToSingle(new Span<byte>(pos, 4));
            pos += sizeof(float);
        }

        return new ReplayDatasetEntry
        {
            Class = (ReplayDatasetClass)header->Class,
            ReplayTokens = new ReplayTokens
            {
                Tokens = data
            }
        };
    }

    byte[] GetEntry(int index)
    {
        DatasetHeaderEntry* header = _headerEntries + index;
        byte* start = _fileStart + header->Start;
        byte[] data = new byte[header->Length];

        Marshal.Copy((nint)start, data, 0, data.Length);

        return data;
    }

    void WriteEntry(int index, byte[] data)
    {
        DatasetHeaderEntry* header = _headerEntries + index;
        byte* start = _fileStart + header->Start;

        Marshal.Copy(data, 0, (nint)start, data.Length);
    }

    static ulong CreateFNVHash(byte[] data)
    {
        //const ulong fnvOffset = 0xcbf29ce484222325;
        const ulong fnvPrime = 0x100000001b3;

        ulong hash = 0;

        foreach (byte b in data)
        {
            hash ^= b;
            hash *= fnvPrime;
        }

        return hash & 0xFFFFFFFFFFFFFFFF;
    }
}