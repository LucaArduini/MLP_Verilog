#include <iostream>
#include <string>
#include <algorithm> // For std::reverse, if needed for endianness visualization
#include <bitset>    // For std::bitset

#include "cnl/include/cnl/all.h" // Or specific CNL headers

using fixed_point_32 = cnl::scaled_integer<int32_t, cnl::power<-16>>;

// Function to print the binary representation of an integer type
template <typename T>
std::string to_binary_string(T val) {
    // Ensure we're working with the correct number of bits for the type
    return std::bitset<sizeof(T) * 8>(val).to_string();
}

int main() {
    fixed_point_32 val1 = fixed_point_32(1.0f);
    fixed_point_32 val2 = fixed_point_32(-0.89f);
    fixed_point_32 val3 = fixed_point_32(32767.9f); // Close to max positive
    fixed_point_32 val4 = fixed_point_32(-32768.0f);   // Min negative

    int32_t rep1 = static_cast<int32_t>(val1);
    int32_t rep2 = static_cast<int32_t>(val2);
    int32_t rep3 = static_cast<int32_t>(val3);
    int32_t rep4 = static_cast<int32_t>(val4);

    std::cout << "fixed: " << val1 << std::endl;
    std::cout << "Value: " << static_cast<double>(val1) << std::endl;
    std::cout << "  Rep: " << rep1 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep1) << std::endl;
    // Example: S  IIIIIIIIIIIIIII FFFFFFFFFFFFFFFF (S=Sign, I=Integer, F=Fractional)
    // For 1.0 (S15.16): 0 000000000000001 0000000000000000
    // Rep: 1 * 2^16 = 65536
    // Bin: 00000000000000010000000000000000

    std::cout << "fixed: " << val2 << std::endl;
    std::cout << "Value: " << static_cast<double>(val2) << std::endl;
    std::cout << "  Rep: " << rep2 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep2) << std::endl;
    // For -0.5 (S15.16):
    // Rep: -0.5 * 2^16 = -32768
    // Bin (two's complement): 11111111111111111000000000000000 (which is -32768 for int32_t)

    std::cout << "fixed: " << val3 << std::endl;
    std::cout << "Value: " << static_cast<double>(val3) << std::endl;
    std::cout << "  Rep: " << rep3 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep3) << std::endl;
    // For 32767.9999... (max positive for S15.16)
    // Rep: (2^15 - 2^-16) * 2^16 = 2^31 - 1 = 2147483647
    // Bin: 01111111111111111111111111111111

    std::cout << "fixed: " << val4 << std::endl;
    std::cout << "Value: " << static_cast<double>(val4) << std::endl;
    std::cout << "  Rep: " << rep4 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep4) << std::endl;
    // For -32768.0 (min negative for S15.16)
    // Rep: -32768.0 * 2^16 = -2147483648
    // Bin: 10000000000000000000000000000000
    using wide_fixed_point = cnl::scaled_integer<int64_t, cnl::power<-2 * 16>>;

    fixed_point_32 a, b;
    a = fixed_point_32(0.867919921875f);
    b = fixed_point_32(1.0f);
    // Multiply with a wider type for intermediate result
    wide_fixed_point result = static_cast<wide_fixed_point>(a) * static_cast<wide_fixed_point>(b);

    // Convert back (with rounding if desired)
    fixed_point_32 final_result = static_cast<fixed_point_32>(result);

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "wide static_cast: " << static_cast<wide_fixed_point>(a) << std::endl;
    std::cout << "wide static_cast: " << static_cast<wide_fixed_point>(b) << std::endl;
    std::cout << "a * b: " << a * b << std::endl;
    std::cout << "Result: " << result << std::endl;
    int32_t raw = cnl::unwrap(a);
    std::bitset<32> bits(raw);
    std::cout << "Raw bits a: " << bits << '\n';
    int32_t rawb = cnl::unwrap(b);
    std::bitset<32> bitsb(rawb);
    std::cout << "Raw bits b: " << bitsb << '\n';
    int32_t rawr = cnl::unwrap(a*b);
    std::bitset<32> bitsr(rawr);
    std::cout << "Raw bits res: " << bitsr << '\n';
    int32_t rawext = cnl::unwrap(static_cast<wide_fixed_point>(a));
    std::bitset<64> bitsext(rawext);
    std::cout << "Raw bits ext a: " << bitsext << '\n';
    int32_t rawextb = cnl::unwrap(static_cast<wide_fixed_point>(b));
    std::bitset<64> bitsextb(rawextb);
    std::cout << "Raw bits ext a: " << bitsextb << '\n';
    auto rawres = cnl::unwrap(a*b);
    std::bitset<32> bitsres(rawres);
    std::cout << "Raw bits res: " << bitsres << '\n';


    return 0;
}