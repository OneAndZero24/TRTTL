#include "../include/trttl.h"
#include <NvInfer.h>
#include <iostream>
#include <cassert>
#include <memory>
#include <vector>

using namespace trttl;

// Test Case for LinearLayer initialization
void testLinearLayerInitialization() {
    std::vector<float> weights(50, 0.1f);
    std::vector<float> biases(5, 0.0f);
    LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT> layer(weights, biases);
    auto paramDims = layer.calcParamDims();
    std::cout << std::get<0>(paramDims).d[1] << " " << std::get<0>(paramDims).d[2] << " " << std::get<1>(paramDims).d[2] << std::endl;

    std::cout << "LinearLayer Initialization Test Passed!" << std::endl;
}

// Test Case for addToNetwork method in LinearLayer
void testLinearLayerAddToNetwork() {
    DefaultLogger logger;

    std::vector<float> weights(50, 0.1f);
    std::vector<float> biases(5, 0.0f);
    LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT> layer(weights, biases);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    trt_types::Network* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims3{1, 1, 10});

    trt_types::Tensor* output_tensor = layer.addToNetwork(network, input);
    network->markOutput(*output_tensor);
    assert(output_tensor != nullptr && "Output tensor should not be null.");

    std::cout << "LinearLayer AddToNetwork Test Passed!" << std::endl;

    delete network;
    delete builder;
}

// Test Case for Sequential initialization
void testSequentialInitialization() {
    std::vector<float> weights1(50, 0.1f);
    std::vector<float> biases1(5, 0.0f);
    std::vector<float> weights2(10, 0.1f);
    std::vector<float> biases2(2, 0.0f);
    LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT> layer1(weights1, biases1);
    LinearLayer<1, trt_types::Dims{2, {1, 5}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT> layer2(weights2, biases2);

    Sequential<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT,
        LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT>,
        LinearLayer<1, trt_types::Dims{2, {1, 5}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT>
        > seq(layer1, layer2);

    std::cout << "Sequential Initialization Test Passed!" << std::endl;
}

// Test Case for Sequential module combining multiple layers
void testSequentialAddToNetwork() {
    DefaultLogger logger;

    std::vector<float> weights1(50, 0.1f);
    std::vector<float> biases1(5, 0.0f);
    std::vector<float> weights2(10, 0.1f);
    std::vector<float> biases2(2, 0.0f);
    LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT> layer1(weights1, biases1);
    LinearLayer<1, trt_types::Dims{2, {1, 5}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT> layer2(weights2, biases2);

    Sequential<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT,
        LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT>,
        LinearLayer<1, trt_types::Dims{2, {1, 5}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT>
        > seq(layer1, layer2);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    trt_types::Network* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims3{1, 1, 10});

    trt_types::Tensor* output_tensor = seq.addToNetwork(network, input);
    network->markOutput(*output_tensor);
    assert(output_tensor != nullptr && "Output tensor should not be null.");

    std::cout << "Sequential AddToNetwork Test Passed!" << std::endl;

    delete network;
    delete builder;
}

// Test Case for ActivationLayer initialization
void testActivationLayerInitialization() {
    ActivationLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::DataType::kFLOAT, trt_types::ActivationType::kRELU> layer;
    static_assert(layer.activation_type == trt_types::ActivationType::kRELU, "Activation type mismatch.");
    std::cout << "ActivationLayer Initialization Test Passed!" << std::endl;
}

// Test Case for addToNetwork method in ActivationLayer
void testActivationLayerAddToNetwork() {
    DefaultLogger logger;

    ActivationLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::DataType::kFLOAT, trt_types::ActivationType::kRELU> layer;

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    trt_types::Network* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims{2, {1, 10}});

    trt_types::Tensor* output_tensor = layer.addToNetwork(network, input);
    network->markOutput(*output_tensor);

    assert(output_tensor != nullptr && "Output tensor should not be null.");

    std::cout << "ActivationLayer AddToNetwork Test Passed!" << std::endl;

    delete network;
    delete builder;
}

// Test Case for SoftmaxLayer initialization
void testSoftmaxLayerInitialization() {
    SoftmaxLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::DataType::kFLOAT> layer;
    std::cout << "SoftmaxLayer Initialization Test Passed!" << std::endl;
}

// Test Case for addToNetwork method in SoftmaxLayer
void testSoftmaxLayerAddToNetwork() {
    DefaultLogger logger;

    SoftmaxLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::DataType::kFLOAT> layer;

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    trt_types::Network* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims{2, {1, 10}});
    
    trt_types::Tensor* output_tensor = layer.addToNetwork(network, input);
    network->markOutput(*output_tensor);

    assert(output_tensor != nullptr && "Output tensor should not be null.");

    std::cout << "SoftmaxLayer AddToNetwork Test Passed!" << std::endl;

    delete network;
    delete builder;
}


// Test Case for creating and building a TensorRT engine
void testTensorRTEngine() {
    DefaultLogger logger;

    Sequential<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT,
        LinearLayer<1, trt_types::Dims{2, {1, 10}}, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT>,
        ActivationLayer<1, trt_types::Dims{2, {1, 5}}, trt_types::DataType::kFLOAT, trt_types::ActivationType::kRELU>,
        LinearLayer<1, trt_types::Dims{2, {1, 5}}, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT>,
        SoftmaxLayer<1, trt_types::Dims{2, {1, 2}}, trt_types::DataType::kFLOAT>
        > seq;

    trttl::Network network(logger, seq);

    auto buffer = network.serialize();

    std::cout << "TensorRT Engine Test Passed!" << std::endl;
}

int main() {
    try {
        testLinearLayerInitialization();
        testLinearLayerAddToNetwork();
        testSequentialInitialization();
        testSequentialAddToNetwork();
        testActivationLayerInitialization();
        testActivationLayerAddToNetwork();
        testSoftmaxLayerInitialization();
        testSoftmaxLayerAddToNetwork();
        testTensorRTEngine();

        std::cout << "All Tests Passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
