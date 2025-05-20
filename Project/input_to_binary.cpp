#include <iostream>                         // for std::cout, std::cerr
#include <fstream>                          // for std::ofstream
#include <climits>                          // for CHAR_BIT
#include <cstdint>                          // for std::int16_t

#include "cnl/include/cnl/all.h"            // for cnl::scaled_integer

const int TOTAL_BITS = 16;
const int FRACTIONAL_BITS = 8;
const int INTEGER_BITS = TOTAL_BITS - FRACTIONAL_BITS;

using fixed_point_16 = cnl::scaled_integer<std::int16_t, cnl::power<-FRACTIONAL_BITS>>;

//////////////////////////////////////////////////////////////////////////////////////////

const int NUM_SET_INPUTS = 1;               // Number of sets of inputs
const int INPUTS_PER_SET = 2;               // Number of inputs per set

// Funzione per stampare i bit di un valore fixed-point_16
// Accetta direttamente un float e fa la conversione internamente.
void print_fixed_point_bits_to_file(std::ofstream& outFile, float float_value) {

    fixed_point_16 fixed_val(float_value);
    const unsigned char* byte_ptr = static_cast<const unsigned char*>(static_cast<const void*>(&fixed_val));

    // Stampiamo i byte in ordine inverso (MSB-first su little-endian)
    for (int byte_idx = sizeof(fixed_val) - 1; byte_idx >= 0; --byte_idx) {
        for (int bit_pos = CHAR_BIT - 1; bit_pos >= 0; --bit_pos) {
            outFile << ((byte_ptr[byte_idx] >> bit_pos) & 1);
        }
    }
    outFile << std::endl;
}

int main(){
    float values[NUM_SET_INPUTS*INPUTS_PER_SET];

    values[0] = 1.5;
    values[1] = -0.5;

    std::ofstream outFile("inputs_bin.txt");

    if (!outFile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return 1;
    }

    for (int i = 0; i < NUM_SET_INPUTS; ++i) {
        if(i>0)
            outFile << std::endl;   // New line between sets

        for (int j = 0; j < INPUTS_PER_SET; ++j)
            print_fixed_point_bits_to_file(outFile, values[i * INPUTS_PER_SET + j]);
    }

    outFile.close();
    std::cout << "Binary representation of inputs saved to inputs_bin.txt" << std::endl;
    return 0;
}