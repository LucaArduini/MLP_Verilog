#include <iostream>

#include <cnl/all.h>

int main() {
    using fixed_32_16 = cnl::scaled_integer<int32_t, cnl::power<-16>>; // 8 bits total, 4 fractional bits

    fixed_32_16 a = 32767.3; // Maximum value for fixed_8_4
    fixed_8_4 b = 32768;
    fixed_8_4 sum = a + b;

    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "sum = " << sum << std::endl;

    return 0;
}