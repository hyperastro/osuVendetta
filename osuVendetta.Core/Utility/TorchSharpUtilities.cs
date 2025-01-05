using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Utils;
using static TorchSharp.torch;

public static class TorchSharpUtilities
{
    public static string ToShapeString(this Dictionary<string, Tensor> stateDict)
    {
        StringBuilder result = new StringBuilder();

        foreach (KeyValuePair<string, Tensor> pair in stateDict)
            result.AppendLine($"{pair.Key}, {string.Join(", ", pair.Value.shape)}");

        return result.ToString();
    }

    public static string ToFlattenedContentString(this Tensor tensor)
    {
        using Tensor flattened = tensor.flatten();
        using TensorAccessor<float> accessor = flattened.data<float>();
        float[] data = accessor.ToArray();

        return string.Join(", ", data);
    }
}
