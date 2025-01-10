<p align="center"><img src="doc/imgs/logo.png" alt="LOGO" width="200"/></p>

# ***TRTTL*** - TensorRT Template Library
![Build Status](https://img.shields.io/github/actions/workflow/status/OneAndZero24/TRTTL/ci-cd.yml) ![TRT ver](https://img.shields.io/badge/TensorRT_ver.-8.6.1-blue) ![TRT ver](https://img.shields.io/badge/C++-20-purple)
### 🚀 ***Accelerate your TensorRT C++ Development!*** 🚀

Lightweight C++ template library that builds on top of TensorRT C++ API extending it with quality-of-life and safety improvements. Compile-time model architecture definition allows for catching dimension mismatches before execution!

***This library is in early development stages (currently it's just PoC). Features may be missing and issues may occur!***

## Features
- Compile time Module definition API
- Compile time data shape/type checks
- Better developer experience
- Predefined layers
- Flexible logger

## Environment
- TensorRT container 23.05
- TensorRT 8.6.1

Tested on NVIDIA GTX 1060 6GB Pascal Architecture.

## TODO
- Convolution Layer
- Split/Join Layers
- Weights loader
- Examples & Benchmarks

## Commands
**Build Image & Compile**
```
docker buildx build -t *TAG* --platform *PLATFORM* --load .
```

**Compile**
```
RUN mkdir build
RUN cd build
RUN cmake ..
RUN make
```

**Test**
```
ctest
```

**Generate Docs**
After compilation in `build` directory:
```
make docs
```