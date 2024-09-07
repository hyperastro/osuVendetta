using OsuParsers.Decoders;
using OsuParsers.Enums.Replays;
using OsuParsers.Replays;
using OsuParsers.Replays.Objects;
using osuVendetta.Core.AntiCheat;
using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;
using NumSharp;
using osuVendetta.Core.AntiCheat.Data;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Concurrent;
using System.Numerics;

namespace osuVendetta.CLI
{
    internal class Program
    {
        public class AntiCheatModel : IAntiCheatModel
        {
            readonly RunOptions _runOptions;

            InferenceSession? _session;

            public AntiCheatModel(string modelPath)
            {
                _runOptions = new RunOptions();

                byte[] modelData = File.ReadAllBytes(modelPath);
                _session = new InferenceSession(modelData);
            }

            public void Dispose()
            {
                _session?.Dispose();
                _session = null;
            }

            public async Task<AntiCheatResult> RunModelAsync(InputArgs args)
            {
                if (_session is null)
                    throw new ObjectDisposedException(nameof(AntiCheatModel));

                int totalChunks = (int)Math.Ceiling(args.InputData.Length / (double)AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK);
                float probabliltyRelaxTotal = 0f;
                float probabliltyNormalTotal = 0f;

                Parallel.For(0, totalChunks, (chunkIndex, cancelToken) =>
                {
                    try
                    {
                        int start = chunkIndex * AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK;

                        using OrtValue value = OrtValue.CreateTensorValueFromMemory(
                            OrtMemoryInfo.DefaultInstance,
                            new Memory<float>(args.InputData, start, AntiCheatService.TOTAL_FEATURE_SIZE_CHUNK),
                            args.Dimensions);

                        Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>
                        {
                            { "input", value }
                        };

                        using IDisposableReadOnlyCollection<OrtValue> output = _session.Run(_runOptions, inputs, _session.OutputNames);
                        ReadOnlySpan<float> outputData = output[0].GetTensorDataAsSpan<float>();

                        probabliltyRelaxTotal += outputData[0]; 
                        probabliltyNormalTotal += outputData[1]; 
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex);
                    }
                });

                probabliltyRelaxTotal /= totalChunks;
                probabliltyNormalTotal /= totalChunks;

                int predictedClass;

                if (probabliltyRelaxTotal < probabliltyNormalTotal)
                    predictedClass = 0;
                else
                    predictedClass = 1;

                switch ((int)predictedClass)
                {
                    case 0:
                        return AntiCheatResult.Relax(predictedClass.ToString());

                    case 1:
                        return AntiCheatResult.Normal(predictedClass.ToString());

                    default:
                        return AntiCheatResult.Invalid($"Unkown result classname index: {predictedClass}");
                }
            }


            static MaxValueIndex GetMaxValueIndex(OrtValue value)
            {
                int maxIndex = -1;
                float maxValue = float.MinValue;

                ReadOnlySpan<float> output = value.GetTensorDataAsSpan<float>();

                for (int i = 0; i < output.Length; i++)
                {
                    float outputValue = output[i];

                    if (outputValue > maxValue)
                    {
                        maxValue = outputValue;
                        maxIndex = i;
                    }
                }

                return new MaxValueIndex(maxIndex, maxValue);
            }
        }

        public class AntiCheatModelProvider : IAntiCheatModelProvider
        {
            readonly string _modelPath = @$"192x2.onnx";

            public async Task<IAntiCheatModel> LoadModelAsync(AntiCheatModelProviderArgs args)
            {
                return new AntiCheatModel(_modelPath);
            }
        }

        static List<ReplayFrame> CreateFrames(string parsedReplayPath)
        {
            List<ReplayFrame> frames = new List<ReplayFrame>();

            int time = 0;
            foreach (string line in File.ReadAllLines(parsedReplayPath))
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

            return frames;
        }

        public record class AntiCheatProviderArgs() : AntiCheatModelProviderArgs;


        static async void TestRun()
        {
            AntiCheatModelProvider provider = new AntiCheatModelProvider();
            AntiCheatService service = new AntiCheatService(provider);
            IAntiCheatModel model = await provider.LoadModelAsync(new AntiCheatProviderArgs());

            //string replayFolder = Path.Combine(Environment.CurrentDirectory, "replays/");

            List<string> files = Directory.EnumerateFiles("replays", "*.txt", SearchOption.AllDirectories).ToList();

            foreach (string file in files)
            {
                List<ReplayFrame> replayFrames = CreateFrames(file);
                float[] inputData = service.ProcessReplayTokensNew(replayFrames);
                long[] dimensions = service.CreateModelDimensionsFor(inputData);

                AntiCheatResult result = await model.RunModelAsync(new InputArgs(inputData, dimensions));
                
                Console.WriteLine($"Result for {file}:\t{result.Type} ({result.Message})");
            }
        }

        static void Main(string[] args)
        {
            TestRun();
            return;

            string replayPath = @"REPLAYDATA_0c4ba34868da87ff45bd3626c86f3cd7.txt";
            string processedPath = @"REPLAYDATA_0c4ba34868da87ff45bd3626c86f3cd7/";

            List<ReplayFrame> frames = CreateFrames(replayPath);

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
