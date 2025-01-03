cmake_minimum_required(VERSION 3.14)
project(trttl VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)      # C++20

# TensorRT
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# CUDA
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(main main.cpp)
target_link_libraries(main nvinfer)     # TensorRT
target_link_libraries(main cudart)      # CUDA Runtime

add_definitions(-O2 -pthread)