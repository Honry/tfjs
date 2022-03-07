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

import {TFLiteNodeModelRunner} from './index';
import * as fs from 'fs';
import * as path from 'path';
import { TFLiteWebModelRunner } from '@tensorflow/tfjs-tflite/dist/types/tflite_web_model_runner';
import '@tensorflow/tfjs-backend-cpu';
import * as jpeg from 'jpeg-js';

//import * as SegfaultHandler from 'segfault-handler';
//SegfaultHandler.registerHandler('crash.log');

describe('interpreter', () => {
  let model: Uint8Array;
  let modelRunner: TFLiteWebModelRunner;
  beforeEach(() => {
    model = fs.readFileSync('./test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite');
    modelRunner = new TFLiteNodeModelRunner(model, { threads: 4 });
  });

  it('has input tensors', () => {
    const inputs = modelRunner.getInputs();
    expect(inputs.length).toEqual(1);
  });

  it('gets data from input tensor', () => {
    const input = modelRunner.getInputs()[0];
    const data = input.data();
    expect(data).toBeDefined();
  });

  it('sets input tensor data', () => {
    const input = modelRunner.getInputs()[0];

    const data = input.data();
    data.set([1,2,3]);
  });

  it('runs infer', () => {
    let outputs = modelRunner.getOutputs();
    modelRunner.infer();
    expect(outputs[0].data()).toBeDefined();
  });

  it('returns the same reference for each getInputs() call', () => {
    expect(modelRunner.getInputs()).toEqual(modelRunner.getInputs());
  });

  it('returns the same reference for each getOutputs() call', () => {
    expect(modelRunner.getOutputs()).toEqual(modelRunner.getOutputs());
  });

  it('returns the same reference for each TensorInfo data() call', () => {
    const input = modelRunner.getInputs()[0];
    const output = modelRunner.getOutputs()[0];
    expect(input.data()).toEqual(input.data());
    expect(output.data()).toEqual(output.data());
  });

  it('gets input tensor name', () => {
    const input = modelRunner.getInputs()[0];
    expect(input.name).toEqual('map/TensorArrayStack/TensorArrayGatherV3');
  });

  it('gets output tensor name', () => {
    const output = modelRunner.getOutputs()[0];
    expect(output.name).toEqual('prediction');
  });

  it('gets input tensor id', () => {
    const input = modelRunner.getInputs()[0];
    expect(input.id).toEqual(0);
  });
});

// function getParrot(): Uint8Array {
//   const parrotJpeg = jpeg.decode(
//     fs.readFileSync('./test_data/parrot-small.jpg'));

//   const {width, height, data} = parrotJpeg;
//   const parrot = new Uint8Array(width * height * 3);
//   let offset = 0;  // offset into original data
//   for (let i = 0; i < parrot.length; i += 3) {
//     parrot[i] = data[offset];
//     parrot[i + 1] = data[offset + 1];
//     parrot[i + 2] = data[offset + 2];

//     offset += 4;
//   }
//   return parrot;
// }

function getMaxIndex(data: ArrayLike<number>) {
  let max = 0;
  let maxIndex = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > max) {
      max = data[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

// describe('model', () => {
//   let model: Uint8Array;
//   let modelRunner: TFLiteWebModelRunner;
//   let parrot: Uint8Array;
//   let labels: string[];

//   beforeEach(() => {
//     model = fs.readFileSync('./test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite');
//     modelRunner = new TFLiteNodeModelRunner(model, { threads: 4 });
//     parrot = getParrot();
//     labels = fs.readFileSync('./test_data/inat_bird_labels.txt', 'utf-8').split('\n');
//   });

//   it('runs a model', () => {
//     const input = modelRunner.getInputs()[0];
//     input.data().set(parrot);
//     modelRunner.infer();
//     const output = modelRunner.getOutputs()[0];
//     const maxIndex = getMaxIndex(output.data());
//     const label = labels[maxIndex];
//     console.log('exact label: ', label);
//     expect(label).toEqual('Ara macao (Scarlet Macaw)');
//   });
// });

describe('float32 support', () => {
  let model: Uint8Array;
  let modelRunner: TFLiteWebModelRunner;
  let red: Float32Array;
  let labels: string[];

  beforeEach(() => {
    model = fs.readFileSync('./test_data/teachable_machine_float.tflite');
    modelRunner = new TFLiteNodeModelRunner(model, {});
    labels = ['class1', 'class2'];

    red = new Float32Array(224 * 224 * 3);
    for (let i = 0; i < red.length; i++) {
      if (i % 3 === 0) {
        red[i] = 255;
      }
    }
  });

  it('model input is a Float32Array', () => {
    const input = modelRunner.getInputs()[0];
    expect(input.data() instanceof Float32Array).toBeTruthy();
  });

  it('model output is a Float32Array', () => {
    const output = modelRunner.getOutputs()[0];
    expect(output.data() instanceof Float32Array).toBeTruthy();
  });

  it('runs a model with float32 input', () => {
    const input = modelRunner.getInputs()[0];
    input.data().set(red);
    modelRunner.infer();
    const output = modelRunner.getOutputs()[0];
    const maxIndex = getMaxIndex(output.data());
    const label = labels[maxIndex];
    expect(label).toEqual('class2');
  });
});

describe('delegate support', () => {
  let model: Uint8Array;
  let modelRunner: TFLiteWebModelRunner;
  // let wine: Float32Array;
  let labels: string[];

  beforeEach(() => {
    console.log('000000000000000000000000')
    model = fs.readFileSync('./test_data/mobilenetv2.tflite');
    console.log('001')
    labels = fs.readFileSync('./test_data/mobilenetv2_labels.txt', 'utf-8').split('\n');
    console.log(labels[0]);
    let delegatePath = path.join(process.cwd(), 'tmp_delegate_dlls', 'webnn_external_delegate.dll');
    console.log(delegatePath);
    modelRunner = new TFLiteNodeModelRunner(model, {
      threads: 4,
      delegate: {
      //   //path: './tmp_delegate_dlls/libedgetpu.1.dylib',
      //   // path: '/Users/matthew/Projects/tfjs/tfjs-tflite-node/tmp_delegate_dlls/libedgetpu.1.dylib',
        path: delegatePath
      },
    });
    console.log('002');
  });

  it('runs a model with the webnn delegate', () => {
    console.log('1111111111111111111')
    const input = modelRunner.getInputs()[0];
    const inputBuffer = input.data();
    const wineJpeg = jpeg.decode(
      fs.readFileSync('./test_data/wine.jpeg'));
    console.log('12222222222222222')
    const { width, height, data } = wineJpeg;
    const inputData = new Float32Array(width * height * 3);
    let pixelIndex = 0;
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        const valStartIndex = pixelIndex * 4;
        const inputIndex = pixelIndex * 3;
        inputData[inputIndex] = (data[valStartIndex] - 127.5) / 127.5;
        inputData[inputIndex + 1] = (data[valStartIndex + 1] - 127.5) / 127.5;
        inputData[inputIndex + 2] = (data[valStartIndex + 2] - 127.5) / 127.5;
        pixelIndex += 1;
      }
    }
    console.log("input data length: ", inputData.length)
    inputBuffer.set(inputData);
    const start = performance.now();
    modelRunner.infer();
    const inferTime = (performance.now() - start).toFixed(2);
    const output = modelRunner.getOutputs()[0];
    console.log("output name: ", output.name);
    console.log("output shape: ", output.shape);
    console.log("output data type: ", output.dataType);
    console.log("Infer time: ", inferTime);
    const result = Array.from(output.data());
    result.shift();  // Remove the first logit which is the background noise.
    console.log(result.length);
    const sortedResult = result
      .map((logit, i) => {
        return { i, logit };
      })
      .sort((a, b) => b.logit - a.logit);
    // Show result.
    const classIndex = sortedResult[0].i;
    const score = sortedResult[0].logit;
    console.log(`classIndex: ${labels[classIndex]}, score: ${score}`);
  });
});
// describe('delegate support', () => {
//   let model: Uint8Array;
//   let modelRunner: TFLiteWebModelRunner;
//   let parrot: Uint8Array;
//   let labels: string[];

//   beforeEach(() => {
//     model = fs.readFileSync('./test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite');
//     modelRunner = new TFLiteNodeModelRunner(model, {
//       threads: 4,
//       delegate: {
//         //path: './tmp_delegate_dlls/libedgetpu.1.dylib',
//         path: '/Users/matthew/Projects/tfjs/tfjs-tflite-node/tmp_delegate_dlls/libedgetpu.1.dylib',
//       },
//     });

//     parrot = getParrot();
//     labels = fs.readFileSync('./test_data/inat_bird_labels.txt', 'utf-8').split('\n');
//   });

//   it('runs a model with the coral delegate', () => {
//     const input = modelRunner.getInputs()[0];
//     input.data().set(parrot);
//     modelRunner.infer();
//     const output = modelRunner.getOutputs()[0];
//     const maxIndex = getMaxIndex(output.data());
//     const label = labels[maxIndex];
//     expect(label).toEqual('Ara macao (Scarlet Macaw)');
//   });
// });
