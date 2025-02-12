using OsuParsers.Replays;
using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
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

public record class DatasetScalerValues(
    DatasetScaler StandardDeviation,
    DatasetScaler StandardMean);

[StructLayout(LayoutKind.Sequential, Pack = 4)]
public record struct DatasetScaler(
    float DeltaTime,
    float X,
    float Y,
    float DeltaX,
    float DeltaY)
{
    public static implicit operator ScalerValues(DatasetScaler scaler)
    {
        return new ScalerValues
        {
            DeltaTime = scaler.DeltaTime,
            X = scaler.X,
            Y = scaler.Y,
            DeltaX = scaler.DeltaX,
            DeltaY = scaler.DeltaY
        };
    }

    public static implicit operator DatasetScaler(ScalerValues scaler)
    {
        return new DatasetScaler(scaler.DeltaTime, scaler.X, scaler.Y, scaler.DeltaX, scaler.DeltaY);
    }
}

public unsafe class DatasetArchive : IDisposable
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    record struct DatasetHeader(
        int TotalEntries,
        DatasetScaler StandardDeviation,
        DatasetScaler StandardMean);

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    record struct DatasetHeaderEntry(
        int Class,
        long Start,
        int Length);

    class DatasetContentAccessor
    {
        public required DatasetStepAccessor* Content { get; init; }
        public required long Length { get; init; }
        public required ReplayDatasetClass Class { get; init; }
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    record struct DatasetStepAccessor(
        float DeltaTime,
        float X,
        float Y,
        float DeltaX,
        float DeltaY);

    public bool IsDisposed { get; private set; }
    public int Count => _header->TotalEntries;

    readonly MemoryMappedFile _mmf;
    readonly MemoryMappedViewAccessor _accessor;
    readonly byte* _fileStart;

    readonly DatasetHeader* _header;
    readonly DatasetHeaderEntry* _headerEntries;
    readonly byte* _content;

    public FileInfo File { get; init; }
    public DatasetScalerValues ScalerValues { get; private set; }

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

        ScalerValues = new DatasetScalerValues(_header->StandardDeviation, _header->StandardMean);
        File = file;
    }

    public void Dispose()
    {
        if (IsDisposed)
            return;

        _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        _accessor.Dispose();
        _mmf.Dispose();

        IsDisposed = true;
    }

    public ReplayDatasetEntry this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);

            return GetDatasetEntry(index);
        }
    }

    public void ExportContentInfo(FileInfo destTxtFile)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

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

    /// <summary>
    /// Creates a new dataset, scaler values will be temporary set to 1 during this progress
    /// <para>Afterwards you can load the archive and call <see cref="UpdateScalerValues(ref AntiCheatModelConfig)"/></para>
    /// </summary>
    /// <param name="file"></param>
    /// <param name="datasets"></param>
    /// <param name="replayProcessor"></param>
    /// <param name="progress"></param>
    public static void Create(FileInfo file, Dictionary<ReplayDatasetClass, List<FileInfo>> datasets,
        IReplayProcessor replayProcessor, IProgress<double>? progress = null, IProgress<double>? progressScalers = null)
    {
        if (file.Exists)
            file.Delete();

        DatasetScalerValues oldScalers = new DatasetScalerValues(
            replayProcessor.AntiCheatModelConfig.StandardDeviation,
            replayProcessor.AntiCheatModelConfig.StandardMean);

        try
        {
            // temporarily reset scaler values
            replayProcessor.AntiCheatModelConfig.StandardDeviation = new ScalerValues
            {
                DeltaTime = 1,
                X = 1,
                Y = 1,
                DeltaX = 1,
                DeltaY = 1,
            };

            replayProcessor.AntiCheatModelConfig.StandardMean = new ScalerValues
            {
                DeltaTime = 0,
                X = 0,
                Y = 0,
                DeltaX = 0,
                DeltaY = 0,
            };

            int totalReplays = datasets.Sum(d => d.Value.Count);
            long contentOffset = sizeof(DatasetHeader) + sizeof(DatasetHeaderEntry) * totalReplays;

            using (FileStream archiveStream = file.Create())
            {
                DatasetHeader header = new DatasetHeader(totalReplays, default, default);
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
                            byte[]? tokens = GetTokenDataFromText(replayStream, replayProcessor.AntiCheatModelConfig);

                            if (tokens is null)
                                continue;

                            tokenData = tokens;
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
                        progress?.Report(processedReplays);
                    }
                }
            }

            using DatasetArchive archive = Load(file);
            archive.UpdateScalerValues(null, progressScalers);

            System.IO.File.WriteAllText($"{file.FullName}.scalers.txt", string.Format(
                System.Globalization.CultureInfo.InvariantCulture,
                "Standard Mean: {0}, {1}, {2}, {3}, {4}\nStandard Deviation: {5}, {6}, {7}, {8}, {9}",
                archive.ScalerValues.StandardMean.DeltaTime,
                archive.ScalerValues.StandardMean.X,
                archive.ScalerValues.StandardMean.Y,
                archive.ScalerValues.StandardMean.DeltaX,
                archive.ScalerValues.StandardMean.DeltaY,
                archive.ScalerValues.StandardDeviation.DeltaTime,
                archive.ScalerValues.StandardDeviation.X,
                archive.ScalerValues.StandardDeviation.Y,
                archive.ScalerValues.StandardDeviation.DeltaX,
                archive.ScalerValues.StandardDeviation.DeltaY));
        }
        finally
        {
            replayProcessor.AntiCheatModelConfig.StandardDeviation = oldScalers.StandardDeviation;
            replayProcessor.AntiCheatModelConfig.StandardMean = oldScalers.StandardMean;
        }
    }

    void RemoveNormalization(DatasetScalerValues scalers)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        for (int i = 0; i < Count; i++)
        {
            DatasetContentAccessor accessor = GetEntry(i);

            for (int j = 0; j < accessor.Length; j++)
            {
                DatasetStepAccessor* step = accessor.Content + j;

                step->DeltaTime = ReplayProcessor.DeNormalize(step->DeltaTime, scalers.StandardMean.DeltaTime, scalers.StandardDeviation.DeltaTime);
                step->X = ReplayProcessor.DeNormalize(step->X, scalers.StandardMean.X, scalers.StandardDeviation.X);
                step->Y = ReplayProcessor.DeNormalize(step->Y, scalers.StandardMean.Y, scalers.StandardDeviation.Y);
                step->DeltaX = ReplayProcessor.DeNormalize(step->DeltaX, scalers.StandardMean.DeltaX, scalers.StandardDeviation.DeltaX);
                step->DeltaY = ReplayProcessor.DeNormalize(step->DeltaY, scalers.StandardMean.DeltaY, scalers.StandardDeviation.DeltaY);
            }
        }
    }

    void ApplyNormalization(DatasetScalerValues scalers)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        for (int i = 0; i < Count; i++)
        {
            DatasetContentAccessor accessor = GetEntry(i);

            for (int j = 0; j < accessor.Length; j++)
            {
                DatasetStepAccessor* step = accessor.Content + j;

                step->DeltaTime = ReplayProcessor.Normalize(step->DeltaTime, scalers.StandardMean.DeltaTime, scalers.StandardDeviation.DeltaTime);
                step->X = ReplayProcessor.Normalize(step->X, scalers.StandardMean.X, scalers.StandardDeviation.X);
                step->Y = ReplayProcessor.Normalize(step->Y, scalers.StandardMean.Y, scalers.StandardDeviation.Y);
                step->DeltaX = ReplayProcessor.Normalize(step->DeltaX, scalers.StandardMean.DeltaX, scalers.StandardDeviation.DeltaX);
                step->DeltaY = ReplayProcessor.Normalize(step->DeltaY, scalers.StandardMean.DeltaY, scalers.StandardDeviation.DeltaY);
            }
        }
    }


    /// <summary>
    /// Generates new scaler values for the current dataset
    /// </summary>
    /// <param name="config"></param>
    /// <returns></returns>
    void UpdateScalerValues(DatasetScalerValues? oldScalers = null, IProgress<double>? progress = null)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);

        if (oldScalers is not null)
            RemoveNormalization(oldScalers);

        double progressValue = 0;

        float totalDeltaTime = 0;
        float totalX = 0;
        float totalY = 0;
        float totalDeltaX = 0;
        float totalDeltaY = 0;
        long totalSteps = 0;

        for (int i = 0; i < Count; i++, progressValue++)
        {
            DatasetContentAccessor content = GetEntry(i);

            for (int j = 0; j < content.Length; j++, totalSteps++)
            {
                DatasetStepAccessor* step = content.Content + j;

                totalDeltaTime += step->DeltaTime;
                totalX += step->X;
                totalY += step->Y;
                totalDeltaX += step->DeltaX;
                totalDeltaY += step->DeltaY;
            }

            progress?.Report(progressValue);
        }

        float meanDeltaTime = totalDeltaTime / totalSteps;
        float meanX = totalX / totalSteps;
        float meanY = totalY / totalSteps;
        float meanDeltaX = totalDeltaX / totalSteps;
        float meanDeltaY = totalDeltaY / totalSteps;

        float totalDevDeltaTime = 0;
        float totalDevX = 0;
        float totalDevY = 0;
        float totalDevDeltaX = 0;
        float totalDevDeltaY = 0;

        for (int i = 0; i < Count; i++, progressValue++)
        {
            DatasetContentAccessor content = GetEntry(i);

            for (int j = 0; j < content.Length; j++, totalSteps++)
            {
                DatasetStepAccessor* step = content.Content + j;

                float devDeltaTime = step->DeltaTime - meanDeltaTime;
                float devX = step->X - meanX;
                float devY = step->Y - meanY;
                float devDeltaX = step->DeltaX - meanX;
                float devDeltaY = step->DeltaY - meanY;

                totalDevDeltaTime += devDeltaTime * devDeltaTime;
                totalDevX += devX * devX;
                totalDevY += devY * devY;
                totalDevDeltaX += devDeltaX * devDeltaX;
                totalDevDeltaY += devDeltaY * devDeltaY;
            }

            progress?.Report(progressValue);
        }

        totalDevDeltaTime /= totalSteps - 1;
        totalDevX /= totalSteps - 1;
        totalDevY /= totalSteps - 1;
        totalDevDeltaX /= totalSteps - 1;
        totalDevDeltaY /= totalSteps - 1;

        _header->StandardDeviation = new DatasetScaler(
            (float)Math.Sqrt(totalDevDeltaTime),
            (float)Math.Sqrt(totalDevX),
            (float)Math.Sqrt(totalDevY),
            (float)Math.Sqrt(totalDevDeltaX),
            (float)Math.Sqrt(totalDevDeltaY));

        _header->StandardMean = new DatasetScaler(
            meanDeltaTime,
            meanX,
            meanY,
            meanDeltaX,
            meanDeltaY);

        ScalerValues = new DatasetScalerValues(_header->StandardDeviation, _header->StandardMean);

        ApplyNormalization(ScalerValues);
    }

    /// <summary>
    /// Adds the input as a float to the list, the input gets normalized
    /// <para>if the normalized or input value is NaN it will be replaced by 0</para>
    /// </summary>
    /// <param name="values"></param>
    /// <param name="input"></param>
    /// <param name="scalerStd"></param>
    /// <param name="scalerMean"></param>
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

            if (float.IsNaN(value))
                value = 0;

            values.Add(value);
        }
    }

    /// <summary>
    /// Gets the token data from the old replay processor in form of text
    /// </summary>
    static byte[]? GetTokenDataFromText(Stream replayStream, AntiCheatModelConfig config)
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
                config.StandardDeviation.DeltaTime,
                config.StandardMean.DeltaTime);
            AddIgnoreNan(tempTokens, split[1],
                config.StandardDeviation.X,
                config.StandardMean.X);
            AddIgnoreNan(tempTokens, split[2],
                config.StandardDeviation.Y,
                config.StandardMean.Y);
            AddIgnoreNan(tempTokens, split[3],
                config.StandardDeviation.DeltaX,
                config.StandardMean.DeltaX);
            AddIgnoreNan(tempTokens, split[4],
                config.StandardDeviation.DeltaY,
                config.StandardMean.DeltaY);

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
            return null;

        return tempTokenData.ToArray();
    }

    ReplayDatasetEntry GetDatasetEntry(int index)
    {
        DatasetContentAccessor content = GetEntry(index);
        float[] data = new float[content.Length * sizeof(DatasetStepAccessor) / sizeof(float)];

        Marshal.Copy((nint)content.Content, data, 0, data.Length);

        return new ReplayDatasetEntry
        {
            Class = content.Class,
            ReplayTokens = new ReplayTokens
            {
                Tokens = data
            }
        };
    }

    DatasetContentAccessor GetEntry(int index)
    {
        DatasetHeaderEntry* header = _headerEntries + index;
        byte* start = _fileStart + header->Start;

        return new DatasetContentAccessor
        {
            Content = (DatasetStepAccessor*)start,
            Length = header->Length / sizeof(DatasetStepAccessor),
            Class = (ReplayDatasetClass)header->Class,
        };
    }

    /// <summary>
    /// Generates a FNV hash out of a byte sequence
    /// </summary>
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