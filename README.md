<p align="center"><img src="doc/imgs/logo.png" alt="LOGO" width="200"/></p>

# ***TRTTL*** - TensorRT Template Library
![Build Status](https://img.shields.io/github/actions/workflow/status/OneAndZero24/TRTTL/ci-cd.yml) ![TRT ver](https://img.shields.io/badge/TensorRT_ver.-8.6.1-blue)
### 🚀 ***Accelerate your TensorRT C++ Development!*** 🚀

Lightweight C++ template library that builds on top of TensorRT C++ API extending it with quality-of-life and safety improvements. Compile-time model architecture definition allows for catching dimension mismatches before execution!

***This library is in early development stages. Features may be missing and issues may occur!***

## Features
- Flexible logger
- Compile time Module definition API
- Compile time data shape/type checks

## Environment
- TensorRT container 23.05
- TensorRT 8.6.1

Tested on NVIDIA GTX 1060 6GB Pascal Architecture.

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