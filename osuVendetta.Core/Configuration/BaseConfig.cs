using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace osuVendetta.Core.Configuration;

public abstract class BaseConfig
{
    public static readonly string ConfigDirectory = "Config";
    public static string BasePath = Environment.CurrentDirectory;

    [JsonIgnore]
    public string Identifier => GetType().Name;

    public static void Save(BaseConfig config)
    {
        string path = GetConfigPath(config.Identifier);
        string json = config.ToString();

        File.WriteAllText(path, json);
    }

    public static TConfig Load<TConfig>()
        where TConfig : BaseConfig
    {
        string path = GetConfigPath(nameof(TConfig));

        if (!File.Exists(path))
        {
            TConfig config = Activator.CreateInstance<TConfig>();

            Save(config);

            return config;
        }

        string json = File.ReadAllText(path);

        return JsonSerializer.Deserialize<TConfig>(json) ??
            throw new InvalidOperationException("Unable to load config from json");
    }

    public override string ToString()
    {
        object obj = this;

        return JsonSerializer.Serialize(obj, new JsonSerializerOptions
        {
            WriteIndented = true,
        });
    }


    static string GetConfigPath(string configIdentifier)
    {
        string basePath = Path.Combine(Environment.CurrentDirectory, ConfigDirectory);

        if (!Directory.Exists(basePath))
            Directory.CreateDirectory(basePath);

        return Path.Combine(Environment.CurrentDirectory, ConfigDirectory, $"{configIdentifier}.json");
    }

}
