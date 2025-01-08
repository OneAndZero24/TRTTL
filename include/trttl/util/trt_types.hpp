#ifndef TRT_TYPES_HPP
#define TRT_TYPES_HPP

#include <NvInfer.h>
#include <algorithm>

namespace trttl {
    namespace trt_types {
        using Severity = nvinfer1::ILogger::Severity;
        using Dims = nvinfer1::Dims;
        using Dims3 = nvinfer1::Dims3;
        using Dims4 = nvinfer1::Dims4;
        using DataType = nvinfer1::DataType;
        using Tensor = nvinfer1::ITensor;
        using Network = nvinfer1::INetworkDefinition;
        using Weights = nvinfer1::Weights;
        using MatrixOperation = nvinfer1::MatrixOperation;
        using ElementWiseOperation = nvinfer1::ElementWiseOperation;
        using ActivationType = nvinfer1::ActivationType;
        using Builder = nvinfer1::IBuilder;
        using BuilderConf = nvinfer1::IBuilderConfig;
        using Memory = nvinfer1::IHostMemory;
    } // trt_types namespace

    constexpr bool operator==(const trt_types::Dims& lhs, const trt_types::Dims& rhs) {
        return (lhs.nbDims == rhs.nbDims) && std::ranges::equal(lhs.d, rhs.d);
    }

    int32_t dimVolume(const trt_types::Dims& dim) {
        int32_t r = 1;
        for(auto i = 0; i < dim.ndDims; ++i)
            r *= d[i];
        return r;
    }

} // trttl namespace
#endif //TRT_TYPES_HPP