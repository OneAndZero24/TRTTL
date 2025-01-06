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
    LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT> layer(weights, biases);
    auto paramDims = layer.calcParamDims();
    std::cout << std::get<0>(paramDims).d[1] << " " << std::get<0>(paramDims).d[2] << " " << std::get<1>(paramDims).d[2] << std::endl;
    assert(50 == std::get<0>(paramDims).d[1] * std::get<0>(paramDims).d[2]);
    assert(5 == std::get<1>(paramDims).d[2]);

    std::cout << "LinearLayer Initialization Test Passed!" << std::endl;
}

// Test Case for addToNetwork method in LinearLayer
void testLinearLayerAddToNetwork() {
    DefaultLogger logger;

    std::vector<float> weights(50, 0.1f);
    std::vector<float> biases(5, 0.0f);
    LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT> layer(weights, biases);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    trt_types::Network* network = builder->createNetworkV2(0U);
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims{1, 10});

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
    std::vector<float> weights2(50, 0.1f);
    std::vector<float> biases2(5, 0.0f);
    LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT> layer1(weights1, biases1);
    LinearLayer<trt_types::Dims{1, 5}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT> layer2(weights2, biases2);

    Sequential<trt_types::Dims{1, 10}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT,
        LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT>,
        LinearLayer<trt_types::Dims{1, 5}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT>
        > seq(layer1, layer2);

    std::cout << "Sequential Initialization Test Passed!" << std::endl;
}

// Test Case for Sequential module combining multiple layers
void testSequentialAddToNetwork() {
    DefaultLogger logger;

    std::vector<float> weights1(50, 0.1f);
    std::vector<float> biases1(5, 0.0f);
    std::vector<float> weights2(50, 0.1f);
    std::vector<float> biases2(5, 0.0f);
    LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT> layer1(weights1, biases1);
    LinearLayer<trt_types::Dims{1, 5}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT> layer2(weights2, biases2);

    Sequential<trt_types::Dims{1, 10}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT,
        LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT>,
        LinearLayer<trt_types::Dims{1, 5}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT>
        > seq(layer1, layer2);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    trt_types::Network* network = builder->createNetworkV2(0U);
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims{1, 10});

    trt_types::Tensor* output_tensor = seq.addToNetwork(network, input);
    network->markOutput(*output_tensor);
    assert(output_tensor != nullptr && "Output tensor should not be null.");

    std::cout << "Sequential AddToNetwork Test Passed!" << std::endl;

    delete network;
    delete builder;
}

// Test Case for creating and building a TensorRT engine
void testTensorRTNetworkAndEngine() {
    DefaultLogger logger;

    std::vector<float> weights1(50, 0.1f);
    std::vector<float> biases1(5, 0.0f);
    std::vector<float> weights2(50, 0.1f);
    std::vector<float> biases2(5, 0.0f);
    LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT> layer1(weights1, biases1);
    LinearLayer<trt_types::Dims{1, 5}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT> layer2(weights2, biases2);

    Sequential<trt_types::Dims{1, 10}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT,
        LinearLayer<trt_types::Dims{1, 10}, trt_types::Dims{1, 5}, trt_types::DataType::kFLOAT>,
        LinearLayer<trt_types::Dims{1, 5}, trt_types::Dims{1, 2}, trt_types::DataType::kFLOAT>
        > seq(layer1, layer2);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    trt_types::Network* network = builder->createNetworkV2(0);
    auto input = network->addInput("input", trt_types::DataType::kFLOAT, trt_types::Dims{1, 10});

    trt_types::Tensor* output_tensor = seq.addToNetwork(network, input);
    network->markOutput(*output_tensor);

    nvinfer1::IHostMemory* buffer{nullptr};
    buffer = builder->buildSerializedNetwork(*network, *config);

    std::cout << "TensorRT Engine Test Passed!" << std::endl;

    delete buffer;
    delete network;
    delete config;
    delete builder;
}

int main() {
    try {
        testLinearLayerInitialization();
        testLinearLayerAddToNetwork();
        testSequentialInitialization();
        testSequentialAddToNetwork();
        testTensorRTNetworkAndEngine();

        std::cout << "All Tests Passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
