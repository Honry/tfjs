"use strict";
var Module = {};
var initializedJS = false;
var pendingNotifiedProxyingQueues = [];
function threadPrintErr() {
  var text = Array.prototype.slice.call(arguments).join(" ");
  console.error(text);
}
function threadAlert() {
  var text = Array.prototype.slice.call(arguments).join(" ");
  postMessage({
    cmd: "alert",
    text: text,
    threadId: Module["_pthread_self"](),
  });
}
var err = threadPrintErr;
self.alert = threadAlert;
Module["instantiateWasm"] = (info, receiveInstance) => {
  var instance = new WebAssembly.Instance(Module["wasmModule"], info);
  receiveInstance(instance);
  Module["wasmModule"] = null;
  return instance.exports;
};
self.onunhandledrejection = (e) => {
  throw e.reason ?? e;
};
self.onmessage = (e) => {
  try {
    if (e.data.cmd === "load") {
      Module["wasmModule"] = e.data.wasmModule;
      Module["wasmMemory"] = e.data.wasmMemory;
      Module["buffer"] = Module["wasmMemory"].buffer;
      Module["ENVIRONMENT_IS_PTHREAD"] = true;
      if (typeof e.data.urlOrBlob == "string") {
        importScripts(e.data.urlOrBlob);
      } else {
        var objectUrl = URL.createObjectURL(e.data.urlOrBlob);
        importScripts(objectUrl);
        URL.revokeObjectURL(objectUrl);
      }
      tflite_model_runner_ModuleFactory(Module).then(function (instance) {
        Module = instance;
      });
    } else if (e.data.cmd === "run") {
      Module["__performance_now_clock_drift"] = performance.now() - e.data.time;
      Module["__emscripten_thread_init"](e.data.pthread_ptr, 0, 0, 1);
      Module["establishStackSpace"]();
      Module["PThread"].receiveObjectTransfer(e.data);
      Module["PThread"].threadInitTLS();
      if (!initializedJS) {
        Module["__embind_initialize_bindings"]();
        pendingNotifiedProxyingQueues.forEach((queue) => {
          Module["executeNotifiedProxyingQueue"](queue);
        });
        pendingNotifiedProxyingQueues = [];
        initializedJS = true;
      }
      try {
        Module["invokeEntryPoint"](e.data.start_routine, e.data.arg);
      } catch (ex) {
        if (ex != "unwind") {
          if (ex instanceof Module["ExitStatus"]) {
            if (Module["keepRuntimeAlive"]()) {
            } else {
              Module["__emscripten_thread_exit"](ex.status);
            }
          } else {
            throw ex;
          }
        }
      }
    } else if (e.data.cmd === "cancel") {
      if (Module["_pthread_self"]()) {
        Module["__emscripten_thread_exit"](-1);
      }
    } else if (e.data.target === "setimmediate") {
    } else if (e.data.cmd === "processProxyingQueue") {
      if (initializedJS) {
        Module["executeNotifiedProxyingQueue"](e.data.queue);
      } else {
        pendingNotifiedProxyingQueues.push(e.data.queue);
      }
    } else if (e.data.cmd) {
      err("worker.js received unknown command " + e.data.cmd);
      err(e.data);
    }
  } catch (ex) {
    if (Module["__emscripten_thread_crashed"]) {
      Module["__emscripten_thread_crashed"]();
    }
    throw ex;
  }
};
