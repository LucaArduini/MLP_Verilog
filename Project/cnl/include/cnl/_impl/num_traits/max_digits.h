
//          Copyright John McFarlane 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_NUM_TRAITS_MAX_DIGITS_H)
#define CNL_IMPL_NUM_TRAITS_MAX_DIGITS_H

#include "../cstdint/types.h"
#include "../numbers/signedness.h"
#include "digits.h"
#include "is_composite.h"
#include "rep_of.h"

#include <concepts>

namespace cnl {
    namespace _impl {
        // max_digits_fundamental
        template<bool Signed>
        inline constexpr int max_digits_fundamental = 0;

        template<>
        inline constexpr auto max_digits_fundamental<true> = digits_v<intmax_t>;

        template<>
        inline constexpr auto max_digits_fundamental<false> = digits_v<uintmax_t>;

        // max_digits
        template<typename T>
        inline constexpr int max_digits = 0;

        template<std::integral T>
        inline constexpr auto max_digits<T> = max_digits_fundamental<numbers::signedness_v<T>>;

        template<typename T>
        requires is_composite<T>::value inline constexpr auto max_digits<T> = max_digits<rep_of_t<T>>;
    }
}

#endif  // CNL_IMPL_NUM_TRAITS_MAX_DIGITS_H
