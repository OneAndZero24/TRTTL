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

    trt_types::Builder* b_ptr;
    trt_types::BuilderConf* c_ptr;
    trt_types::Network* network;

    void build() {
        c_ptr = b_ptr->createBuilderConfig();
        network = b_ptr->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        int32_t size = module.in_shape.nbDims+1;
        std::vector<int32_t> d(size, 0);
        d[0] = module.batch_size;
        std::copy(module.in_shape.d, module.in_shape.d+size, d.begin()+1);
        auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims{size, *d.data()});

        trt_types::Tensor* output_tensor = module.addToNetwork(network, input);
        network->markOutput(*output_tensor);
    }

public: 
    Network(trt_types::Builder* builder) : b_ptr(builder) {
        build();
    }

    Network(trt_types::Builder* builder, M m) : module(m),  b_ptr(builder) {
        build();
    }

    ~Network() {
        delete network;
        delete c_ptr;
    }

    std::unique_ptr<trt_types::Memory> serialize() {
        std::unique_ptr<trt_types::Memory> buffer(b_ptr->buildSerializedNetwork(*network, *c_ptr));
        return buffer;
    }
};

} // trttl namespace
#endif // UTILS_Hł