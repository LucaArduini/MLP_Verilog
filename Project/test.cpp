#include <iostream>
#include <string>
#include <algorithm> // For std::reverse, if needed for endianness visualization
#include <bitset>    // For std::bitset

#include <cnl/all.h> // Or specific CNL headers

using fixed_point_32 = cnl::scaled_integer<int32_t, cnl::power<-16>>;

// Function to print the binary representation of an integer type
template <typename T>
std::string to_binary_string(T val) {
    // Ensure we're working with the correct number of bits for the type
    return std::bitset<sizeof(T) * 8>(val).to_string();
}

int main() {
    fixed_point_32 val1 = fixed_point_32(1.0);
    fixed_point_32 val2 = fixed_point_32(-0.5);
    fixed_point_32 val3 = fixed_point_32(32767.9999); // Close to max positive
    fixed_point_32 val4 = fixed_point_32(-32768.0);   // Min negative

    int32_t rep1 = static_cast<int32_t>(val1);
    int32_t rep2 = static_cast<int32_t>(val2);
    int32_t rep3 = static_cast<int32_t>(val3);
    int32_t rep4 = static_cast<int32_t>(val4);

    std::cout << "Value: " << static_cast<double>(val1) << std::endl;
    std::cout << "  Rep: " << rep1 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep1) << std::endl;
    // Example: S  IIIIIIIIIIIIIII FFFFFFFFFFFFFFFF (S=Sign, I=Integer, F=Fractional)
    // For 1.0 (S15.16): 0 000000000000001 0000000000000000
    // Rep: 1 * 2^16 = 65536
    // Bin: 00000000000000010000000000000000

    std::cout << "\nValue: " << static_cast<double>(val2) << std::endl;
    std::cout << "  Rep: " << rep2 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep2) << std::endl;
    // For -0.5 (S15.16):
    // Rep: -0.5 * 2^16 = -32768
    // Bin (two's complement): 11111111111111111000000000000000 (which is -32768 for int32_t)

    std::cout << "\nValue: " << static_cast<double>(val3) << std::endl;
    std::cout << "  Rep: " << rep3 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep3) << std::endl;
    // For 32767.9999... (max positive for S15.16)
    // Rep: (2^15 - 2^-16) * 2^16 = 2^31 - 1 = 2147483647
    // Bin: 01111111111111111111111111111111

    std::cout << "\nValue: " << static_cast<double>(val4) << std::endl;
    std::cout << "  Rep: " << rep4 << std::endl;
    std::cout << "  Bin: " << to_binary_string(rep4) << std::endl;
    // For -32768.0 (min negative for S15.16)
    // Rep: -32768.0 * 2^16 = -2147483648
    // Bin: 10000000000000000000000000000000

    return 0;
}