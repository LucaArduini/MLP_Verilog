
//          Copyright John McFarlane 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// \brief fixed-width integer types equivalent to those in \verbatim<cstdint>\endverbatim

#if !defined(CNL_IMPL_CSTDINT_TYPES_H)
#define CNL_IMPL_CSTDINT_TYPES_H

#include "../config.h"

#include <cstdint>

/// compositional numeric library
namespace cnl {
#if defined(CNL_INT128_ENABLED)
    // to disable 128-bit integer support, #define CNL_USE_INT128=0
    using int128_t = __int128;
    using uint128_t = unsigned __int128;

    using intmax_t = int128_t;
    using uintmax_t = uint128_t;
#else
    using intmax_t = std::intmax_t;
    using uintmax_t = std::uintmax_t;
#endif  // defined(CNL_INT128_ENABLED)
}

#endif  // CNL_IMPL_CSTDINT_TYPES_H
