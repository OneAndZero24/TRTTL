#include "NvInfer.h"
#include <iostream>
#include <memory>


// From TRT C++ API docs
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


// Simple TRT test
int main() {
    try {
        std::cout << "TensorRT Version: " << NV_TENSORRT_MAJOR << "." 
            << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
        auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, 1, 1});
        network->markOutput(*input);

        std::cout << "OK" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}