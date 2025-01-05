#ifndef CEXPR_UTILS_HPP
#define CEXPR_UTILS_HPP

#include <type_traits>

namespace trttl {
    namespace cexpr_utils {
        /*!
        * Returns last param.
        */
        template<typename... Ts>
        struct last_rec;

        template<typename T>
        struct last_rec<T> {
            using type = T; 
        };

        template<typename T1, typename T2, typename... Ts>
        struct last_rec<T1, T2, Ts...> {
            using type = typename last_rec<Ts...>::type;
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