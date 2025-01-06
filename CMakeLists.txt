cmake_minimum_required(VERSION 3.14)
project(trttl VERSION 0.2.4)

set(CMAKE_CXX_STANDARD 20) # C++20
set(CXX_STANDARD_REQUIRED ON)

# TensorRT
set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu/)
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu/)
# CUDA
set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
set(CUDA_LIB_DIR /usr/local/cuda/lib64)

include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR} ${CUDA_LIB_DIR})

add_definitions(-O2 -pthread)

# Enable CTest
enable_testing()

# Collect all test source files
file(GLOB TEST_SOURCES test/test*.cpp)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE) # Get name without extension
    add_executable(${TEST_NAME} ${TEST_SOURCE})              # Create executable
    target_link_libraries(${TEST_NAME} nvinfer cudart)       # Link TensorRT and CUDA
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})         # Register the test
endforeach()

# CTest config
set(CTEST_OUTPUT_ON_FAILURE ON)
set(CTEST_PARALLEL_LEVEL 4)

# Find Doxygen
set(DOXYGEN_EXECUTABLE /doxygen/doxygen-1.13.1/bin/doxygen)
find_package(Doxygen REQUIRED)

set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/include)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/doc)
set(DOXYGEN_CONFIG_FILE ${CMAKE_BINARY_DIR}/Doxyfile)

configure_file(${CMAKE_SOURCE_DIR}/Doxyfile ${DOXYGEN_CONFIG_FILE} @ONLY)

# Add a custom target for documentation
add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)