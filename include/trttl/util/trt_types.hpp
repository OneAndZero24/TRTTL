#ifndef TRT_TYPES_HPP
#define TRT_TYPES_HPP

#include <NvInfer.h>

namespace trttl {
    namespace trt_types {
        using Severity = nvinfer1::ILogger::Severity;
        using Dims = nvinfer1::Dims;
        using DataType = nvinfer1::DataType;
    } // trt_types namespace
} // trttl namespace
#endif //TRT_TYPES_HPP