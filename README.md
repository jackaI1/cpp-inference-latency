# C++ inference latency

I built a small image model benchmark because I wanted to see whether making a model smaller actually makes its predictions faster without losing accuracy. It generates a fixed set of images, runs the same images through two versions of a model, and reports how often each version is right and how long it takes. The images contain horizontal or vertical bars; the model learns to tell them apart.

The two versions are a normal floating-point model and a quantized model, which stores its learned weights as smaller integers. A smaller model is not always faster: on my Apple Silicon Mac, both were 100% accurate on 128 generated test images, but the integer version took 0.00213 ms per batch at the median versus 0.00163 ms for the float version. These tiny timings are illustrative, not a prediction for larger models or other computers.

## Try it

You need Python 3, CMake, a C++17 compiler, and an [ONNX Runtime C++ release](https://github.com/microsoft/onnxruntime/releases) for your operating system. ONNX is a portable format for trained models; ONNX Runtime runs them. Download and unpack the 1.20.1 release, then point `ONNXRUNTIME_ROOT` at that directory.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install torch numpy onnx onnxruntime
python scripts/export.py
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build
ctest --test-dir build --output-on-failure
build/benchmark artifacts/float.onnx artifacts/images.txt 16 5 100 artifacts/float.csv
build/benchmark artifacts/int8.onnx artifacts/images.txt 16 5 100 artifacts/int8.csv
python scripts/check.py artifacts/float.csv
```

The arguments are model, image file, batch size, warmup passes, measured passes, and output CSV. The image count must be divisible by the batch size. The CSV contains the label and two raw scores for each image. The check compares the C++ float scores with PyTorch, the Python library used to train the model, using absolute tolerance `1e-5` and relative tolerance `1e-4`.

## How it works

The export script trains a small fully connected classifier on generated noisy bars and saves a separate, reproducible test set. It exports the float model and uses dynamic weight quantization to produce the integer version. This is a controlled example, not a benchmark of natural photos or a pretrained production model.

The C++ program reads the same test images for each model, warms up the runtime, then measures each batch with a monotonic clock. It reports median (p50) and 95th percentile (p95) batch latency, images per second based on measured inference time, classification accuracy, and peak resident memory for the whole process. Memory includes model loading and inputs; timing excludes file loading and CSV writing. Each run uses one inference thread. Compare runs on the same machine with the same arguments; very small models are sensitive to timer overhead and CPU scheduling.

The percentile unit test checks interpolation and empty input. GitHub Actions builds the C++ program on Linux, exports both models, runs both against 128 images and checks the float output against PyTorch. The local run above had a maximum float score difference of 0.00000143.
