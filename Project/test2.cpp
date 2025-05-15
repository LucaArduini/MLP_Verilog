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

    if (mul > std::numeric_limits<int32_t>::max()) {
        std::cout << "Overflow detected!\n";
        return fixed_point_32{std::numeric_limits<int32_t>::max()};
    } else if (mul < std::numeric_limits<int32_t>::min()) {
        std::cout << "Underflow detected!\n";
        return fixed_point_32{std::numeric_limits<int32_t>::min()};
    }

    std::cout << "raw mul: " << mul << "\n";

    std::bitset<64> bitres(mul);
    std::cout << "Raw bits m: " << bitres << '\n';

    std::bitset<32> bitsres(static_cast<int32_t>(mul));
    std::string bitstr = bitsres.to_string();
    bitstr.insert(16, " "); // Insert a space after the first 16 bits (MSB side)

    std::cout << "Raw bits res: " << bitstr << '\n';

    return cnl::from_rep<fixed_point_32, int32_t>{}(static_cast<int32_t>(mul));
}

fixed_point_32 fixed_point_divide(fixed_point_32 a, fixed_point_32 b) {
    int64_t a_raw = impl::to_rep(a);
    int64_t b_raw = impl::to_rep(b);

    std::cout << "a.raw: " << a_raw << "\n";
    std::cout << "b.raw: " << b_raw << "\n";

    std::bitset<64> bita(a_raw);
    std::cout << "Raw bits a: " << bita << '\n';
    std::bitset<64> bitb(b_raw);
    std::cout << "Raw bits b: " << bitb << '\n';

    int64_t a_raw_shifted = a_raw << fractional_bits;

    int64_t div = (a_raw_shifted / b_raw);

    std::cout << "raw div: " << div << "\n";

    std::bitset<64> bitres(div);
    std::cout << "Raw bits : " << bitres << '\n';

    std::bitset<32> bitsres(static_cast<int32_t>(div));
    std::string bitstr = bitsres.to_string();
    bitstr.insert(16, " "); // Insert a space after the first 16 bits (MSB side)

    std::cout << "Raw bits div: " << bitstr << '\n';

    return cnl::from_rep<fixed_point_32, int32_t>{}(static_cast<int32_t>(div));
}

int main() {
    fixed_point_32 a = fixed_point_32{-10.0f};
    fixed_point_32 b = fixed_point_32{-2.0f};
    fixed_point_32 result = fixed_point_divide(a, b);
    fixed_point_32 result2 = result + fixed_point_32{21.5632f};


    std::cout << "Result (float): " << static_cast<float>(result) << "\n";
    std::cout << "Result (raw bits): " << std::bitset<32>(impl::to_rep(result)) << "\n";
    std::cout << "Result (fixed_point_32): " << result << "\n";
    std::cout << "max between a and b: " << std::max(a, b) << "\n";
    std::cout << "Result2 (float): " << static_cast<float>(result2) << "\n";
    std::cout << "Result2 (raw bits): " << std::bitset<32>(impl::to_rep(result2)) << "\n";
    std::cout << "Result2 (fixed_point_32): " << result2 << "\n";
    std:: cout << "result with overloaded operator: " << a * b << "\n";
}
