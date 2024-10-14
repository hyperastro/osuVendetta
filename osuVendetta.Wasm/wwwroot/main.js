import { dotnet } from './_framework/dotnet.js';

const { setModuleImports, getAssemblyExports, getConfig } = await dotnet
    .withDiagnosticTracing(false)
    .withApplicationArgumentsFromQuery()
    .create();

setModuleImports('main.js', {
    loadModel: (modelPath) => {
        // Placeholder for loading the model
        console.log(`Model loaded from: ${modelPath}`);
    },
    loadConfig: () => {
        // Placeholder for loading the configuration
        const config = JSON.stringify({ /* config object */ });
        console.log('Configuration loaded');
        return config;
    },
    runModel: (inputNames, inputValues, shape) => {
        // Placeholder for running the model with inputs
        console.log('Running model with inputs:', { inputNames, inputValues, shape });
        return JSON.stringify({ /* model result */ });
    }
});

const config = getConfig();
const exports = await getAssemblyExports(config.mainAssemblyName);

// Example usage of C# methods
const initialize = async () => {
    const configJson = exports.osuVendetta.Wasm.Core.LoadConfigJson();
    const modelPath = '/path/to/model.onnx'; // Adjust the model path
    await exports.osuVendetta.Wasm.Core.InitializeAsync(configJson, modelPath);
};

const processReplay = async (replayData, runInParallel = false) => {
    const result = await exports.osuVendetta.Wasm.Core.ProcessReplayAsync(replayData, runInParallel);
    console.log('Replay processed, result:', result);
};

// Example HTML manipulation (optional)
document.getElementById('out').innerHTML = 'osuVendetta WASM initialized';

await dotnet.run();