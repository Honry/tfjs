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

import '@webmachinelearning/webnn-polyfill';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';

import {expose, proxy} from 'comlink';
import {setWasmPath, loadTFLiteModel, TFLiteWebModelRunnerOptions} from '@tensorflow/tfjs-tflite';


export type Predictor = {predict: (data: Float32Array) => Promise<Float32Array>};

const api = {
  async setWebNNPolyfillBackend(deviceType: string): Promise<void> {
    // Initiate webnn-polyfill
    const backend = deviceType == 'gpu' ? 'webgl' : 'wasm';
    console.warn(`WebNN ${deviceType} backend is not supported on this browser, \
fallback to use webnn-polyfill ${backend} backend.`);
    const context = navigator.ml.createContextSync();
    const tf = context.tf;
    await tf.setBackend(backend);
    await tf.ready();
    proxy(this.setWebNNPolyfillBackend);
  },
  setWasmPath: proxy(setWasmPath),
  async loadTFLiteModel(modelPath: string, options: TFLiteWebModelRunnerOptions): Promise<Predictor> {

    const model = await loadTFLiteModel(modelPath, options)

    const wrapped: Predictor = {
      async predict(data) {
        // Hacky hard-coded tensor shape :(
        const prediction = model.predict(tf.tensor(data, [1, 224, 224, 3]));
        return (prediction as tf.Tensor).dataSync() as Float32Array;
      }
    }

    return proxy(wrapped);
  }
}

export type Api = typeof api;

expose(api);
