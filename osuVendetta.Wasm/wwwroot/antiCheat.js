import { dotnet } from './_framework/dotnet.js';

const { setModuleImports, getAssemblyExports, getConfig } = await dotnet
    .withDiagnosticTracing(false)
    .withApplicationArgumentsFromQuery()
    .create();

const config = getConfig();
const exports = await getAssemblyExports(config.mainAssemblyName);

// Create antiCheat object which contains functions exported by WASM
const antiCheat = {
    session = null,

    // Signature: Task InitializeAsync(string config, string modelPath)
    async initialize(configJson, modelPath) {
        await exports.osuVendetta.Wasm.Core.InitializeAsync(configJson, modelPath);
    },

    // Signature: Task<string> ProcessReplayAsync(byte[] replayData, bool runInParallel)
    async processReplay(replayData, runInParallel = false) {
        return await exports.osuVendetta.Wasm.Core.ProcessReplayAsync(replayData, runInParallel);
    }
};

// Set functions to be imported to WASM
setModuleImports('antiCheat.js', {

    // Signature: Task LoadModel(string modelPath);
    loadModel: async (modelPath) => {
        antiCheat.session = await ort.InferenceSession.create(modelPath);
    },

    // Signature: Task<string> LoadConfigJson();
    loadConfig: async () => {
        const config = JSON.stringify({
            // See: osuVendetta.CoreLib.AntiCheat.Data.AntiCheatConfig, naming in web standard
            // probably get this from wwwroot or idk
        });
        return config;
    },

    // Signature: Task<string> RunModel(string[] inputNames, double[] inputValues, int[] shape);
    runModel: async (inputValues, shape) => {
        const feeds = { input: new ort.Tensor('float32', inputValues, shape) };
        const results = await antiCheat.session.run(feeds);
        const output = results.output.data;

        return JSON.stringify({
            relax: output[0],
            normal: output[1]
        });
    }
});


// run the WASM
await dotnet.run();

/*
    1. load this .js file
    2. initialize the antiCheat
    3. process any replay

    TODO: check if conversions of datatypes are done correctly

    onnxruntime: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    ^add this to the same page as this script
*/
