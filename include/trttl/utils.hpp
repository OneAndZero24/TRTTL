#ifndef UTILS_H
#define UTILS_H

#include "util/trt_types.hpp"
#include "modules.hpp"
#include <NvInfer.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace trttl {

/*!
* Utility wrapper around the process of creation and serialization of NNs.
*/
template<DerivedFromModule M>
class Network {
private:
    M module;

    trt_types::Builder* builder;
    trt_types::BuilderConf* config;
    trt_types::Network* network;

    void build(nvinfer1::ILogger &logger) {
        builder = nvinfer1::createInferBuilder(logger);
        config = builder->createBuilderConfig();
        network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        trt_types::Dims in_dims;
        in_dims.nbDims = module.in_shape.nbDims+1;
        in_dims.d[0] = module.batch_size;
        for (auto i = 0; i < module.in_shape.nbDims; ++i)
            in_dims.d[i+1] = module.in_shape.d[i];
        auto input = network->addInput("input", trt_types::DataType::kFLOAT, in_dims);

        trt_types::Tensor* output_tensor = module.addToNetwork(network, input);
        network->markOutput(*output_tensor);
    }

public: 
    Network(nvinfer1::ILogger& log) {
        build(log);
    }

    Network(nvinfer1::ILogger& log, M m) : module(m) {
        build(log);
    }

    ~Network() {
        delete network;
        delete config;
        delete builder;
    }

    std::unique_ptr<trt_types::Memory> serialize() {
        std::unique_ptr<trt_types::Memory> buffer(builder->buildSerializedNetwork(*network, *config));
        return buffer;
    }
};

} // trttl namespace
#endif // UTILS_Hł