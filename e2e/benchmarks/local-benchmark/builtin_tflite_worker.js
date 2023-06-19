/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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

importScripts('https://cdn.jsdelivr.net/npm/comlink@latest/dist/umd/comlink.js');
importScripts('./builtin-tfjs-tflite/tflite_model_runner_cc_simd.js');

const tfliteWorkerAPI = {
  async loadTFLiteModel(modelPath, options) {

    // Load WASM module and model.
    const [module, modelArrayBuffer] = await Promise.all([
      tflite_model_runner_ModuleFactory(),
      (await fetch(modelPath)).arrayBuffer(),
    ]);
    // Load WASM module and model.
    const modelBytes = new Uint8Array(modelArrayBuffer);
    const offset = module._malloc(modelBytes.length);
    module.HEAPU8.set(modelBytes, offset);

    // Create model runner.
    const model = module.TFLiteWebModelRunner.CreateFromBufferAndOptions(
      offset,
      modelBytes.length,
      {
        numThreads: 1,
        enableWebNNDelegate: options.useWebnn,
        webNNDevicePreference: 2,
        webNNNumThreads: options.numThreads,
      }
    );
    const modelRunner = model.value();
    if (!model.ok()) {
      throw new Error(`Failed to create TFLiteWebModelRunner: ${modelRunner.errorMessage()}`);
    }
    const inputs = callAndDelete(modelRunner.GetInputs(), (results) => convertCppVectorToArray(results));
    const outputs = callAndDelete(modelRunner.GetOutputs(), (results) => convertCppVectorToArray(results));
    const wrapped = {
      inputs: inputs,
      getProfilingResults() {
        // Builtin delegate doesn't support profiling.
        throw new Error(`Builtin tfjs-tflite doesn't support profiling.`);
      },
      cleanUp() {
        modelRunner.delete();
      },
      async predict(inputData) {
        if (!inputData[0].length) {
          // Single input, move it into an arrary
          inputData = [inputData];
        }
        for (let i = 0; i < inputs.length; i++) {
          const inputBuffer = inputs[i].data();
          inputBuffer.set(inputData[i]);
        }
        modelRunner.Infer();

        // Get output data.
        for (let output of outputs) {
          output.data();
        }
        // We encourage the user to process output data in the worker thread
        // rather than posting output data to main thread directly, as
        // this would bring much overhead if the output size is huge.
        // From this perspective, we don't post output data to main thread
        // in this benchmark.
        return "OK";
      },
    };

    return Comlink.proxy(wrapped);
  },
};

Comlink.expose(tfliteWorkerAPI);


// Helper functions.

// Converts the given c++ vector to a JS array.
function convertCppVectorToArray(vector) {
  if (vector == null) return [];

  const result = [];
  for (let i = 0; i < vector.size(); i++) {
    const item = vector.get(i);
    result.push(item);
  }
  return result;
}


// Calls the given function with the given deletable argument, ensuring that
// the argument gets deleted afterwards (even if the function throws an error).
function callAndDelete(arg, func) {
  try {
    return func(arg);
  } finally {
    if (arg != null) arg.delete();
  }
}