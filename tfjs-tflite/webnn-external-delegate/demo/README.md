# TFLite WebNN external delegate demo

_WORK IN PROGRESS_

This demo classifies image and your webcam feed into pre-defined categories by using the
[`tfjs-tflite`][tfjs-tflite] package and the [MobileNet V2][MobileNet V2]
model, accelerates the inference via the WebNN external delegate.

Here is the [live demo][live demo] you can try out. If the browser does not have a native
implementation of the [WebNN API][webnn] available, this demo will fallback to use the
[webnn-polyfill][webnn-polyfill].

## Run the demo locally

Build the dependencies.

```sh
$ yarn build-deps
$ yarn
```

Run the demo locally
```sh
$ yarn watch
```

[tfjs-tflite]: https://www.npmjs.com/package/@tensorflow/tfjs-tflite
[MobileNet V2]: https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1
[live demo]: 
[webnn]: https://webmachinelearning.github.io/webnn
[webnn-polyfill]: https://www.npmjs.com/package/@webmachinelearning/webnn-polyfill