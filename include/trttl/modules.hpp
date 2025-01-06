#ifndef MODULE_HPP
#define MODULE_HPP

#include "util/cexpr_utils.hpp"
#include "util/trt_types.hpp"
#include <NvInfer.h>
#include <concepts>

namespace trttl {

template<typename Derived, trt_types::Dims in_shape, trt_types::Dims out_shape, trt_types::DataType dt>
class Module {
public:
    Module() {
        static_assert(std::is_member_function_pointer_v<decltype(&Derived::addToNetwork_impl)>, 
                      "Derived must implement addToNetwork_impl().");
    }

    trt_types::Tensor* addToNetwork(){
        return tatic_cast<Derived*>(this)->addToNetwork_impl();
    }

    static constexpr trt_types::Dims in_shape = in_shape;
    static constexpr trt_types::Dims out_shape = out_shape;
    static constexpr trt_types::DataType dt = dt;
};

template<typename T>
concept DerivedFromModule = std::derived_from<T, Module<T, T::in_shape, T::out_shape, T::dt>>;

template<typename... Ms>
struct check_seq;

template<DerivedFromModule M>
struct check_seq<M> : std::true_type {};

template<DerivedFromModule M1, DerivedFromModule M2, DerivedFromModule... Ms>
struct check_seq<M1, M2, Ms...>
    : std::conditional_t<
        (M1::out_shape == M2::in_shape)&&(M1::dt == M2::dt), 
        check_seq<M2, Ms...>, 
        std::false_type> 
    {};

template<trt_types::Dims in_shape, trt_types::Dims out_shape, trt_types::DataType dt, DerivedFromModule M, DerivedFromModule... Ms>
requires (M::in_shape == in_shape && cexpr_utils::last<Ms...>::out_shape == out_shape && check_seq<M, Ms...>::value)
class Sequential : Module<Sequential, in_shape, out_shape, dt> {
public:
    // TODO layers
    // TODO addToNetwork
};

} // trttl namespace
#endif //MODULE_HPP

// TODO
// - DOCS
// - FC
// - Test