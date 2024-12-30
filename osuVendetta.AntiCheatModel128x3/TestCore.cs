using osuVendetta.Core.Anticheat.Data;
using osuVendetta.Core.AntiCheat;
using osuVendetta.Core.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace osuVendetta.AntiCheatModel128x3;

internal static class TestCore
{
    static async Task Main(string[] args)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;

        try
        {
            string pathExportsdModel = "Data/128x3.dat";
            string pathModelWeights = "Data/128x3.bin";
            string pathModelSafetensors = "Data/128x3.safetensors";
            string pathModelStateDict = "Data/state_dict_128x3.dat";

            Safetensors safetensors = Safetensors.Load(pathModelSafetensors);
            Dictionary<string, Tensor> stateDict = safetensors.ToStateDict();

            //foreach ((string key, Tensor tensor) in stateDict)
            //{
            //    Console.WriteLine($"{key} [{string.Join(", ", tensor.shape)}]:");
            //    Console.WriteLine(tensor.ToString(numpy));
            //}

            ////foreach ((string key, SafetensorsJsonHeaderEntry entry) in safetensors.Header)
            ////{
            //    //Console.WriteLine($"{key}:\n{entry}\n----");
            //    //byte[] data = safetensors.GetDataFromHeader(entry);

            //    //StringBuilder sb = new StringBuilder();
            //    //foreach (byte b in data)
            //    //    sb.Append($"{b}, ");
            //    //sb.Remove(sb.Length - 2, 2);

            //    //Console.WriteLine($"Data:\n{sb}");

            //    //break;
            ////}

            AntiCheatModel128x3 model = new AntiCheatModel128x3();
            model.load_state_dict(stateDict);

            //var dict = torch.load(pathExportsdModel);
            //var dict2 = nn.Module.

            //using FieldAccessException.e
            //using FileStream modelStream = File.OpenRead(pathExportsdModel);
            //model.Load(modelStream);

            RunTest(model);
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(ex);
            Console.ForegroundColor = ConsoleColor.Cyan;
        }

        Console.WriteLine("Done");
        await Task.Delay(-1);
    }

    static void RunTest(AntiCheatModel128x3 model)
    {
        float[,,] input = LoadTestInput(model);

        AntiCheatModelResult result = model.RunInference(new Core.Replays.Data.ReplayTokens
        {
            ReplayName = "Test",
            Tokens = input
        });

        StringBuilder segmentBuilder = new StringBuilder();

        for (int i = 0; i < result.Segments.Length; i++)
            segmentBuilder.Append($"{result.Segments[i]}, ");

        segmentBuilder.Remove(segmentBuilder.Length - 2, 2);
        
        Console.WriteLine(segmentBuilder.ToString());
    }

    static float[,,] LoadTestInput(AntiCheatModel128x3 model)
    {
        float[,,] tokens = new float[1, model.Config.StepsPerChunk, model.Config.FeaturesPerStep];

        int steps = 0;
        foreach (string line in File.ReadAllLines("Data/modelInput.csv"))
        {
            string[] lineSplit = line.Split(',');

            for (int i = 0; i < model.Config.FeaturesPerStep; i++)
            {
                if (!float.TryParse(lineSplit[i], out float token))
                {
                    Console.WriteLine($"Unable to parse {lineSplit[i]}");
                    continue;
                }

                tokens[0, steps, i] = token;
            }

            steps++;
        }

        return tokens;
    }
}
