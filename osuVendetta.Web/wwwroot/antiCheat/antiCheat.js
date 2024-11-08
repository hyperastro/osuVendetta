window.antiCheat = {
    load: async function (modelPath) {
        if (!this.session)
            this.session = await ort.InferenceSession.create(modelPath);
    },
    run: async function (inputData, dimensions) {
        const feeds = { input: new ort.Tensor('float32', inputData, dimensions) };
        const results = await this.session.run(feeds);
        const output = results.output.data;

        return output;
    },

    unload: async function () {

    }
}