#include <iostream>
#include "cnl/include/cnl/all.h"


int main() {
    using fixed_8_4 = cnl::scaled_integer<int, cnl::power<-4>>; // 8 bits total, 4 fractional bits

    fixed_8_4 a = 1.5; 
    fixed_8_4 b = 2.25;
    fixed_8_4 sum = a + b;

    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "sum = " << sum << std::endl;

    return 0;
}