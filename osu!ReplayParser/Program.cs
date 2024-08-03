using System.Diagnostics;
using OsuParsers.Decoders;
using OsuParsers.Replays;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays.Objects;
using System.Text;

namespace osu_ReplayParser;

internal class Program
{
    static void Main(string[] args)
    {
        List<FileInfo> files = new List<FileInfo>();
        string? outputDir = null;

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];

            switch (arg)
            {
                case "-out":
                    outputDir = args[++i];            
                    continue;
            }

            if (Directory.Exists(arg))
                files.AddRange(new DirectoryInfo(arg).EnumerateFiles("*.osr"));
            else if (File.Exists(arg))
                files.Add(new FileInfo(arg));
            else
                Console.WriteLine($"Unkown path: {arg}");
        }

        if (string.IsNullOrEmpty(outputDir))
            outputDir = "out/";
        else if (!Directory.Exists(outputDir))
            Directory.CreateDirectory(outputDir);

        Stopwatch sw = new Stopwatch();
        sw.Start();
        Parallel.For(0, files.Count, i => ParseOsr(files[i].FullName, Path.Combine(outputDir, files[i].Name) + ".txt"));
        sw.Stop();

        float msPerReplay = (float)sw.ElapsedMilliseconds / files.Count;

        Console.WriteLine($"Processing {files.Count} files took {sw.ElapsedMilliseconds} ms\n{msPerReplay} ms per replay");
    }

    public static void ParseOsr(string osrPath, string outputPath)
    {
        using FileStream fstream = new FileStream(osrPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        Replay replay = ReplayDecoder.Decode(fstream);

        if (replay.Ruleset != OsuParsers.Enums.Ruleset.Standard)
            return;

        float lastX = replay.ReplayFrames[0].X;
        float lastY = replay.ReplayFrames[0].Y;

        StringBuilder strBuilder = new StringBuilder();

        for (int i = 0; i < replay.ReplayFrames.Count; i++)
        {
            ReplayFrame frame = replay.ReplayFrames[i];

            float xDiff = frame.X - Interlocked.Exchange(ref lastX, frame.X);
            float yDiff = frame.Y - Interlocked.Exchange(ref lastY, frame.Y);

            strBuilder.Append($"{frame.TimeDiff},{frame.X},{frame.Y},{xDiff},{yDiff},{GetKeyString(frame.StandardKeys)}\n");
        }

        File.WriteAllText(outputPath, strBuilder.ToString());
    }

    static string GetKeyString(StandardKeys keys)
    {
        if (keys.HasFlag(StandardKeys.M1))
        {
            if (keys.HasFlag(StandardKeys.M2))
                return "M1M2";
            else
                return "M1";
        }
        else if (keys.HasFlag(StandardKeys.M2))
        {
            return "M2";
        }
        else
        {
            return "None";
        }
    }
}