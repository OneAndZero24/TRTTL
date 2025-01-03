#include "NvInfer.h"
#include <iostream>
#include <vector>
#include <memory>


// From TRT C++ API docs
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


enum class ActivationType {
    NONE,
    RELU
};


template<int InputSize, int OutputSize, ActivationType Activation = ActivationType::NONE>
class LinearLayer {
public:
    LinearLayer() {
        weights.resize(InputSize * OutputSize, 0.1f);
        biases.resize(OutputSize, 0.0f);
    }

    nvinfer1::ITensor* addLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, weights.data(), static_cast<int64_t>(weights.size())};
        nvinfer1::Weights bs{nvinfer1::DataType::kFLOAT, biases.data(), static_cast<int64_t>(biases.size())};

        auto matmul = network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE, *wt, nvinfer1::MatrixOperation::kNONE);
        auto add = network->addElementWise(*matmul->getOutput(0), *bs, nvinfer1::ElementWiseOperation::kSUM);

        nvinfer1::ITensor* output = ad->getOutput(0);

        if constexpr (Activation == ActivationType::RELU) {
            auto relu = network->addActivation(*output, nvinfer1::ActivationType::kRELU);
            output = relu->getOutput(0);
        }
        return output;
    }

private:
    std::vector<float> weights;
    std::vector<float> biases;
};


template<typename... Layers>
class Sequential {
public:
    Sequential() : layers_(std::make_tuple(Layers...)) {}

    nvinfer1::ITensor* addLayersToNetwork(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input) {
        return addLayersToNetworkImpl(network, input, std::index_sequence_for<Layers...>{});
    }

private:
    std::tuple<Layers...> layers_;

    template<std::size_t... Is>
    nvinfer1::ITensor* addLayersToNetworkImpl(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input, std::index_sequence<Is...>) {
        ((input = std::get<Is>(layers_).addLayer(network, input)), ...);
        return input;
    }
};


int main() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, 1, 128});

    Sequential<
        LinearLayer<128, 64, ActivationType::RELU>,
        LinearLayer<64, 32>,
        LinearLayer<32, 10, ActivationType::RELU>
    > model;

    auto output = model.addLayersToNetwork(network, input);
    network->markOutput(*output);

    return 0;
}
