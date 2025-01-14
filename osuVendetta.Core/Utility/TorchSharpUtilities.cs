using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
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
        float[] data = tensor.ToArray<float>();

        return string.Join(", ", data);
    }

    public static T[] ToArray<T>(this Tensor tensor)
        where T : unmanaged
    {
        using Tensor flattened = tensor.flatten();
        using TensorAccessor<T> accessor = flattened.data<T>();

        return accessor.ToArray();
    }

    public static Tensor? DisposeAndReplace(this Tensor? tensor, Tensor? newTensor, bool moveToOuterScope)
    {
        tensor?.Dispose();

        if (moveToOuterScope)
            return newTensor?.MoveToOuterDisposeScope();
        else
            return newTensor;
    }

    public static Tensor? MoveToOutmostScope(this Tensor? tensor)
    {
        if (tensor is null)
            return null;

        for (; ; )
            tensor.MoveToOuterDisposeScope();
    }
}
