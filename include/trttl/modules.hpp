#ifndef MODULE_HPP
#define MODULE_HPP

#include "util/cexpr_utils.hpp"
#include "util/trt_types.hpp"
#include <NvInfer.h>
#include <concepts>
#include <vector>
#include <tuple>

namespace trttl {

/*!
* Analog to PyTorch's `nn.Module` - represents differentiable operations and their compositions.
*
* @tparam Derived - CRTP
* @tparam bs - batch size
* @tparam in - input shape
* @tparam out - out shape
* @tparam dt - data type
*/
template<typename Derived, int32_t bs, trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt>
class Module {
public:
    Module() {
        static_assert(std::is_member_function_pointer_v<decltype(&Derived::addToNetwork_impl)>, 
                      "Derived must implement addToNetwork_impl().");
    }

    /*!
    * Adds module to TRT network definition.
    */
    trt_types::Tensor* addToNetwork(trt_types::Network* network, trt_types::Tensor* data){
        return static_cast<Derived*>(this)->addToNetwork_impl(network, data);
    }

    static constexpr int32_t batch_size = bs;
    static constexpr trt_types::Dims in_shape = in;
    static constexpr trt_types::Dims out_shape = out;
    static constexpr trt_types::DataType data_type = dt;
};

/*!
* Concept for classes derived from Module.
*/
template<typename T>
concept DerivedFromModule = std::derived_from<T, Module<T, T::batch_size, T::in_shape, T::out_shape, T::data_type>>;

/*!
* Checks whether data types/batch sizes in a sequence of modules match. Needed for Sequential.
*/
template<typename... Ms>
struct check_seq;

template<DerivedFromModule M>
struct check_seq<M> : std::true_type {};

template<DerivedFromModule M1, DerivedFromModule M2, DerivedFromModule... Ms>
struct check_seq<M1, M2, Ms...>
    : std::conditional_t<
        (M1::batch_size == M2::batch_size)&&(M1::out_shape == M2::in_shape)&&(M1::data_type == M2::data_type), 
        check_seq<M2, Ms...>, 
        std::false_type> 
    {};

/*!
* Sequential module - allows to compose operations/layers one after another.
* 
* @tparam M - at least one module inside
* @tparam Ms - possibly more
*/
template<int32_t bs, trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt, DerivedFromModule M, DerivedFromModule... Ms>
requires (M::in_shape == in && M::data_type == dt && M::batch_size == bs && cexpr_utils::last<Ms...>::out_shape == out && check_seq<M, Ms...>::value)
class Sequential : public Module<Sequential<bs, in, out, dt, M, Ms...>, bs, in, out, dt> {
private:
    std::tuple<M, Ms...> modules;

public:
    Sequential() : modules() {}
    Sequential(M m, Ms... ms) : modules(m, ms...) {}

    template<DerivedFromModule... Modules>
    static constexpr trt_types::Tensor* addToNetwork_fold(trt_types::Network* network, trt_types::Tensor* data, Modules... modules) {
        ((data = modules.addToNetwork_impl(network, data)), ...);
        return data;
    }

    trt_types::Tensor* addToNetwork_impl(trt_types::Network* network, trt_types::Tensor* data) {
        return std::apply(addToNetwork_fold<M, Ms...>, std::tuple_cat(std::make_tuple(network, data), modules));
    }
};

/*!
* FullyConnected LinearLayer - pretty self-explanatory.
*/
template<int32_t bs, trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt>
requires (in.nbDims == 2 && out.nbDims == 2)
class LinearLayer : public Module<LinearLayer<bs, in, out, dt>, bs, in, out, dt> {
private:
    std::vector<float> w_data;
    std::vector<float> b_data;

public:
    LinearLayer() {
        w_data = std::vector<float>(dimVolume(in)*dimVolume(out), 0.1f);
        b_data = std::vector<float>(dimVolume(out), 0.1f);
    }

    LinearLayer(std::vector<float> &weights, std::vector<float> &biases) {
        w_data = weights;
        b_data = biases;
    }
    
    static auto calcParamDims() {
        return std::make_tuple(trt_types::Dims3{bs, dimVolume(in), dimVolume(out)}, trt_types::Dims3{bs, 1, dimVolume(out)});
    }

    trt_types::Tensor* addToNetwork_impl(trt_types::Network* network, trt_types::Tensor* data) {
        auto paramDims = calcParamDims();

        auto weights = trt_types::Weights{dt, w_data.data(), static_cast<int64_t>(w_data.size())};
        auto w_tensor = network->addConstant(std::get<0>(paramDims), weights)->getOutput(0);
        auto matmul = network->addMatrixMultiply(*data, trt_types::MatrixOperation::kNONE, *w_tensor, trt_types::MatrixOperation::kNONE);

        auto biases = trt_types::Weights{dt, b_data.data(), static_cast<int64_t>(b_data.size())};
        auto b_tensor = network->addConstant(std::get<1>(paramDims), biases)->getOutput(0);
        auto add = network->addElementWise(*matmul->getOutput(0), *b_tensor, trt_types::ElementWiseOperation::kSUM);

        return add->getOutput(0);
    }
};

/*!
* Activation Layer.
*/
template<int32_t bs, trt_types::Dims size, trt_types::DataType dt, trt_types::ActivationType at>
class ActivationLayer : public Module<ActivationLayer<bs, size, dt, at>, bs, size, size, dt> {
public:
    static constexpr trt_types::ActivationType activation_type = at;

    trt_types::Tensor* addToNetwork_impl(trt_types::Network* network, trt_types::Tensor* data) {
        auto activation = network->addActivation(*data, activation_type);
        return activation->getOutput(0);
    }
};

/*!
* Softmax Layer.
*/
template<int32_t bs, trt_types::Dims size, trt_types::DataType dt>
class SoftmaxLayer : public Module<SoftmaxLayer<bs, size, dt>, bs, size, size, dt> {
public:
    trt_types::Tensor* addToNetwork_impl(trt_types::Network* network, trt_types::Tensor* data) {
        auto sm = network->addSoftMax(*data);
        return sm->getOutput(0);
    }
};

/*!
* Template meta-function to compute the `Sequential` module recursively for the MLP.
*/
template<int32_t bs, trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt, int32_t first, int32_t second, int32_t... rest>
struct MLP_Helper {
    using Type = Sequential<bs, in, out, dt,
        LinearLayer<bs, in, trt_types::Dims{2, {1, second}}, dt>,
        ActivationLayer<bs, trt_types::Dims{2, {1, second}}, dt, trt_types::ActivationType::kRELU>,
        typename MLP_Helper<bs, trt_types::Dims{2, {1, second}}, out, dt, second, rest...>::Type>;
};

template<int32_t bs, trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt, int32_t first, int32_t second>
struct MLP_Helper<bs, in, out, dt, first, second> {
    using Type = Sequential<bs, in, out, dt,
        LinearLayer<bs, in, trt_types::Dims{2, {1, second}}, dt>,
        ActivationLayer<bs, trt_types::Dims{2, {1, second}}, dt, trt_types::ActivationType::kRELU>,
        LinearLayer<bs, trt_types::Dims{2, {1, second}}, out, dt>,
        SoftmaxLayer<bs, out, dt>
    >;
};

/*!
* Multi-Layer Percepttron
*/
template<int32_t bs, trt_types::DataType dt, int32_t first, int32_t... hiddenSizes>
class MLP : public MLP_Helper<bs, trt_types::Dims{2, {1, first}}, trt_types::Dims{2, {1, cexpr_utils::last_rec::value(first, hiddenSizes...)}}, dt, first, hiddenSizes...>::Type {
    using Base = typename MLP_Helper<bs, trt_types::Dims{2, {1, first}}, trt_types::Dims{2, {1, cexpr_utils::last_rec::value(first, hiddenSizes...)}},dt, first, hiddenSizes...>::Type;

public:
    MLP() : Base() {}
};

} // trttl namespace
#endif //MODULE_HPP