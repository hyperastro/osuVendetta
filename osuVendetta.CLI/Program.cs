using OsuParsers.Decoders;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.AntiCheat;
using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;
using NumSharp;

namespace osuVendetta.CLI
{
    internal class Program
    {
        static void Main(string[] args)
        {

            string replayPath = @"REPLAYDATA_0c4ba34868da87ff45bd3626c86f3cd7.txt";
            string processedPath = @"REPLAYDATA_0c4ba34868da87ff45bd3626c86f3cd7/";

            List<ReplayFrame> frames = new List<ReplayFrame>();

            int time = 0;
            foreach (string line in File.ReadAllLines(replayPath))
            {
                string[] split = line.Split(',');
                // timediff, x, y, deltax, deltay, key
                ReplayFrame frame = new ReplayFrame
                {
                    TimeDiff = int.Parse(split[0]),
                    X = float.Parse(split[1]),
                    Y = float.Parse(split[2]),
                    StandardKeys = ToKey(split[5])
                };

                time += frame.TimeDiff;
                frame.Time = time;
                frames.Add(frame);
            }

            AntiCheatService acs = new AntiCheatService(null);
            float[] input = acs.ProcessReplayTokensNew(frames);
            List<float> input2 = new List<float>(input.Length);

            for (int i = 1; i <= 39; i++)
            {
                foreach (string line in File.ReadAllLines($"{processedPath}segment_{i}.txt"))
                {
                    string[] split = line.Split(' ');

                    // end of file: "Label: relax"
                    if (split.Length <= 1)
                        break;

                    foreach (string splitEntry in split)
                        input2.Add(float.Parse(splitEntry));
                }
            }

            float avgDiff = 0;
            int avgDiffCount = 0;
            int featureIndex = 0;
            int indexFirstZero = -1;
            int indexFirstZero2 = -1;
            for (int i = 0; i < input2.Count; i++)
            {
                if (featureIndex == 5)
                    featureIndex = -1;

                featureIndex++;

                float a = input[i];
                float b = input2[i];
                float diff = a - b;

                if (indexFirstZero < 0 && a == 0)
                    indexFirstZero = i;
                else if (indexFirstZero >= 0 && a != 0)
                    indexFirstZero = -1;

                if (indexFirstZero2 < 0 && b == 0)
                    indexFirstZero2 = i;
                else if (indexFirstZero2 >= 0 && b != 0)
                    indexFirstZero2 = -1;

                if (diff > 0.001f || diff < -0.001f)
                {
                    avgDiff += diff;
                    avgDiffCount++;

                    if (i < 5000 || i >= input2.Count - 5000)
                        Console.WriteLine($"Mismatch (Index: {i}\tFeature: {featureIndex})\t{a} vs {b}");
                }
            }

            avgDiff /= avgDiffCount;

            Console.WriteLine($"Mismatches: {avgDiffCount}");
            Console.WriteLine($"Average difference:\t{avgDiff:n6}");
            Console.WriteLine($"Length: {input.Length} vs {input2.Count}");
        }

        static StandardKeys ToKey(string value)
        {
            switch (value.ToLower())
            {
                default: 
                    return StandardKeys.None;

                case "m1":
                    return StandardKeys.M1;

                case "m2":
                    return StandardKeys.M2;

                case "m1m2":
                    return StandardKeys.M1 | StandardKeys.M2;

            }
        }
    }
}
