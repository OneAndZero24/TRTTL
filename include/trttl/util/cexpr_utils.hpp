#ifndef CEXPR_UTILS_HPP
#define CEXPR_UTILS_HPP

#include <type_traits>

namespace trttl {
    namespace cexpr_utils {
        /*!
        * Returns last param type/value.
        */
        template <typename... Ts>
        struct last_rec {
            static_assert(sizeof...(Ts) > 0, "Empty parameter pack! Cannot deduce the last type.");
        };

        template<typename T>
        struct last_rec<T> {
            using type = T; 
            static constexpr T value(T last_value) {
                return last_value;
            }
        };

        template<typename T1, typename T2, typename... Ts>
        struct last_rec<T1, T2, Ts...> {
            using type = typename last_rec<T2, Ts...>::type;
            template <typename... Args>
            static constexpr type value(T1, T2, Args... args) {
                return last_rec<T2, Ts...>::value(args...);
            }
        };

        template<typename... Ts>
        using last = typename last_rec<Ts...>::type;

        /*!
        * Converts enum to underlying type.
        */
        template <typename T, typename R = std::underlying_type_t<T>>
        constexpr R to_underlying(T e) noexcept {
            return static_cast<R>(e);
        }
    } // cexpr_utils namespace
} // trttl namespace
#endif //CEXPR_UTILS_HPP