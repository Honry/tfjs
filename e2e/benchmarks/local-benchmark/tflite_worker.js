/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

let modelRunner, tfliteWebNNModel;

const api = {
    loadTfliteWebNNRunner: loadTfliteWebNNRunner,
    getTfliteWebNNModelInputs: getInputs,
    infer: infer,
};

async function loadTfliteWebNNRunner(payload) {
    const { url, hasMTSupport, options } = payload;
    if (hasMTSupport) {
        importScripts('./tflite-support/tflite_model_runner_cc_threaded_simd.js');
    } else {
        importScripts('./tflite-support/tflite_model_runner_cc_simd.js');
    }
    // Load WASM module and model.
    const [module, modelArrayBuffer] = await Promise.all([
        tflite_model_runner_ModuleFactory(),
        (await fetch(url)).arrayBuffer(),
    ]);

    const modelBytes = new Uint8Array(modelArrayBuffer);
    const offset = module._malloc(modelBytes.length);
    module.HEAPU8.set(modelBytes, offset);

    console.log(
        `loadTfliteWebNNRunner ${url} with numThreads = ${options.numThreads}`
    );

    // Create model runner.
    const modelRunnerResult =
        module.TFLiteWebModelRunner.CreateFromBufferAndOptions(
            offset,
            modelBytes.length,
            options
        );
    if (!modelRunnerResult.ok()) {
        throw new Error('Failed to create TFLiteWebModelRunner: ' + modelRunnerResult.errorMessage());
    }
    modelRunner = modelRunnerResult.value();
}

// Serialize the inputs
function getInputs() {
    const inputs = getTfliteWebNNModelInputs();
    return inputs.map(input => ({ dataType: input.dataType, name: input.name, shape: input.shape }));
}

function getTfliteWebNNModelInputs() {
    const inputs = callAndDelete(
        modelRunner.GetInputs(), results => convertCppVectorToArray(results));
    return inputs;
}


function getTfliteWebNNModelOutputs() {
    const outputs = callAndDelete(
        modelRunner.GetOutputs(), results => convertCppVectorToArray(results));
    return outputs;
}


function infer(payload) {
    const { input } = payload;
    const inputs = getTfliteWebNNModelInputs();
    const inputBuffer = inputs[0].data();
    const inputData = new Float32Array(inputBuffer.length);
    const outputs = getTfliteWebNNModelOutputs();
    const output = outputs[0];
    inputData.set(input);
    inputBuffer.set(inputData);
    modelRunner.Infer();
    return output;
}

onmessage = async (message) => {
    const { actionType, payload } = message.data;
    const result = await api[actionType](payload);
    this.postMessage(result);
};

/**
 * Calls the given function with the given deletable argument, ensuring that
 * the argument gets deleted afterwards (even if the function throws an error).
 */
function callAndDelete(arg, func) {
    try {
        return func(arg);
    } finally {
        if (arg != null) arg.delete();
    }
}


/** Converts the given c++ vector to a JS array. */
function convertCppVectorToArray(vector) {
    if (vector == null) return [];

    const result = [];
    for (let i = 0; i < vector.size(); i++) {
        const item = vector.get(i);
        result.push(item);
    }
    return result;
}