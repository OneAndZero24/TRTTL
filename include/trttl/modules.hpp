#ifndef MODULE_HPP
#define MODULE_HPP

#include "util/cexpr_utils.hpp"
#include "util/trt_types.hpp"
#include <NvInfer.h>
#include <concepts>

namespace trttl {

template<typename T>
concept NetworkConstructible = std::constructible_from<T, (nvinfer1::INetworkDefinition*)>;

template<NetworkConstructible Derived, trt_types::Dims in_shape, trt_types::Dims out_shape, trt_types::DataType dt, bool standalone = false>
class Module {
public:
    virtual ~Module() = 0;  // Enforcing abstract class

    virtual ICudaEngine* buildEngine() requires standalone = 0;

    static constexpr trt_types::Dims in_shape = in_shape;
    static constexpr trt_types::Dims out_shape = out_shape;
    static constexpr trt_types::DataType dt = dt;
};

template<typename T>
concept DuckModule = requires {
    {NetworkConstructible<T>};
    { T::in_shape } -> std::convertible_to<trt_types::Dims>;
    { T::out_shape } -> std::convertible_to<trt_types::Dims>;
    { T::dt } -> std::convertible_to<trt_types::DataType>;
};

template <DuckModule M>
struct check_seq : std::true_type {};

template <DuckModule M1, DuckModule M2, DuckModule... Ms>
struct check_seq<M1, M2, Ms...>
    : std::conditional_t<
        (M1::out_shape == M2.in_shape)&&(M1::dt == M2.dt), 
        check_seq<M, Ms...>, 
        std::false_type> 
    {};

template<trt_types::Dims in_shape, trt_types::Dims out_shape, trt_types::DataType dt, bool standalone = false, DuckModule M, DuckModule... Ms>
requires (M::in_shape == in_shape && cexpr_utils::last<Ms...>::out_shape == out_shape && check_seq<M, Ms...>)
class Sequential : Module<Sequential, in_shape, out_shape, dt, standalone> {
    // TODO NetworkDefinition constructor
    // TODO buildEngine
};

} // trttl namespace
#endif //MODULE_HPP

// TODO
// - DOCS
// - check templates online