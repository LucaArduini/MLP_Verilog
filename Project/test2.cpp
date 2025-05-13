#include <cnl/scaled_integer.h>
#include <iostream>
#include <bitset>
#include <limits> // Required for std::numeric_limits
namespace impl = cnl::_impl;

constexpr int fractional_bits = 16;
using fixed_point_32 = cnl::scaled_integer<int32_t, cnl::power<-fractional_bits>>;

fixed_point_32 fixed_point_multiply(fixed_point_32 a, fixed_point_32 b) {
    int64_t a_raw = impl::to_rep(a);
    int64_t b_raw = impl::to_rep(b);

    std::cout << "a.raw: " << a_raw << "\n";
    std::cout << "b.raw: " << b_raw << "\n";

    std::bitset<64> bita(a_raw);
    std::cout << "Raw bits a: " << bita << '\n';
    std::bitset<64> bitb(b_raw);
    std::cout << "Raw bits b: " << bitb << '\n';

    int64_t mul = (a_raw * b_raw) >> fractional_bits;

    std::cout << "raw mul: " << mul << "\n";

    std::bitset<64> bitres(mul);
    std::cout << "Raw bits mul: " << bitres << '\n';

    std::bitset<32> bitsres(static_cast<int32_t>(mul));
    std::cout << "Raw bits res: " << bitsres << '\n';

    return cnl::from_rep<fixed_point_32, int32_t>{}(static_cast<int32_t>(mul));
}

int main() {
    fixed_point_32 a = fixed_point_32{1.0f};
    fixed_point_32 b = fixed_point_32{0.892839f};
    fixed_point_32 result = fixed_point_multiply(a, b);

    std::cout << "Result (float): " << static_cast<float>(result) << "\n";
    std::cout << "Result (raw bits): " << std::bitset<32>(impl::to_rep(result)) << "\n";
}
