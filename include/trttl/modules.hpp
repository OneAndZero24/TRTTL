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
* @tparam in - input shape
* @tparam out - out shape
* @tparam dt - data type
*/
template<typename Derived, trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt>
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

    static constexpr trt_types::Dims in_shape = in;
    static constexpr trt_types::Dims out_shape = out;
    static constexpr trt_types::DataType data_type = dt;
};

/*!
* Concept for classes derived from Module.
*/
template<typename T>
concept DerivedFromModule = std::derived_from<T, Module<T, T::in_shape, T::out_shape, T::data_type>>;

/*!
* Checks whether data types in a sequence of modules match. Needed for Sequential.
*/
template<typename... Ms>
struct check_seq;

template<DerivedFromModule M>
struct check_seq<M> : std::true_type {};

template<DerivedFromModule M1, DerivedFromModule M2, DerivedFromModule... Ms>
struct check_seq<M1, M2, Ms...>
    : std::conditional_t<
        (M1::out_shape == M2::in_shape)&&(M1::data_type == M2::data_type), 
        check_seq<M2, Ms...>, 
        std::false_type> 
    {};

/*!
* Sequential module - allows to compose operations/layers one after another.
* 
* @tparam M - at least one module inside
* @tparam Ms - possibly more
*/
template<trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt, DerivedFromModule M, DerivedFromModule... Ms>
requires (M::in_shape == in && M::data_type == dt && cexpr_utils::last<Ms...>::out_shape == out && check_seq<M, Ms...>::value)
class Sequential : public Module<Sequential<in, out, dt, M, Ms...>, in, out, dt> {
private:
    std::tuple<M, Ms...> modules;

public:
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
template<trt_types::Dims in, trt_types::Dims out, trt_types::DataType dt>
class LinearLayer : public Module<LinearLayer<in, out, dt>, in, out, dt> {
private:
    std::vector<float> w_data;
    std::vector<float> b_data;

public:
    LinearLayer(std::vector<float> &weights, std::vector<float> &biases) {
        w_data = weights;
        b_data = biases;
    }
    
    static auto calcParamDims() {
        return std::make_tuple(trt_types::Dims{3, {1, in.d[0], out.d[0]}}, trt_types::Dims{3, {1, 1, out.d[0]}});
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


} // trttl namespace
#endif //MODULE_HPP