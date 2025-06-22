#include <iostream>
#include <array>
#include <cmath>
#include <fstream>

#include "cnl/include/cnl/all.h"    // For cnl::scaled_integer

namespace impl = cnl::_impl;
// Using namespaces for convenience in this .cpp file
using namespace std;
using namespace cnl;
const int fractional_bits = 8;     // Number of fractional bits for fixed-point representation   

// Define the fixed-point type
using fixed_point_16 = cnl::scaled_integer<int16_t, cnl::power<-fractional_bits>>;
using temp_accumulator = cnl::scaled_integer<int64_t, cnl::power<-fractional_bits*2>>; // 

// //////////////////////////////////////////// //
//             MLP parameters for inference     //
// //////////////////////////////////////////// //
const int n_output = 1;             // Number of outputs
const int n_features = 2;           // Number of input features
const int n_hidden = 4;           // Number of neurons in the hidden layer

// Weights are assumed to be loaded from a file
array<array<fixed_point_16, n_features + 1>, n_hidden> w1;
array<array<fixed_point_16, n_hidden + 1>, n_output> w2;

// Forward propagation variables
array<fixed_point_16, n_features + 1> a0_infer;
array<fixed_point_16, n_hidden> rZ1_infer;
array<fixed_point_16, n_hidden> rA1_infer;
array<fixed_point_16, n_hidden + 1> a1_infer;
array<fixed_point_16, n_output> rZ2_infer;
array<fixed_point_16, n_output> y_pred_infer;


fixed_point_16 MLP_predict_single_sample(const array<fixed_point_16, n_features>& single_x_val) {
    array<fixed_point_16, n_features + 1> local_a0;
    array<fixed_point_16, n_hidden>       local_rZ1;
    array<fixed_point_16, n_hidden>       local_rA1;
    array<fixed_point_16, n_hidden + 1>   local_a1;
    array<fixed_point_16, n_output>       local_rZ2;

    local_a0[0] = 1.0f;
    for (int j = 0; j < n_features; ++j) local_a0[j + 1] = single_x_val[j];

    for (int h = 0; h < n_hidden; ++h) {
        temp_accumulator sum = 0.0;
        for (int f = 0; f < n_features + 1; ++f) {
            sum += static_cast<temp_accumulator>(w1[h][f]) * local_a0[f];
        }
        local_rZ1[h] = static_cast<fixed_point_16>(sum);
    }

    for (int h = 0; h < n_hidden; ++h) local_rA1[h] = max(fixed_point_16{0.0f}, local_rZ1[h]);


    local_a1[0] = 1.0f;
    for (int h = 0; h < n_hidden; ++h) local_a1[h + 1] = local_rA1[h];
    
    temp_accumulator sum_out = 0.0;
    for (int h = 0; h < n_hidden + 1; ++h) { 
        sum_out += static_cast<temp_accumulator>(w2[0][h]) * local_a1[h];
    }
    local_rZ2[0] = static_cast<fixed_point_16>(sum_out);

    return local_rZ2[0];
}


// Function to load weights from a file
bool load_weights(const string& filename_w1, const string& filename_w2) {
    ifstream file_w1(filename_w1);
    ifstream file_w2(filename_w2);

    if (!file_w1.is_open() || !file_w2.is_open()) {
        cerr << "Error: Could not open weight files." << endl;
        return false;
    }

    string line;
    // float value; // Original type
    fixed_point_16 value; // Changed to fixed-point

    // Load w1
    for (int i = 0; i < n_hidden; ++i) {
        
        for (int j = 0; j < n_features + 1; ++j) {
            //If reading from file:
            float temp_val_w1;
            if (!(file_w1 >> temp_val_w1)) {
                cerr << "Error: Could not read weight w1[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w1[i][j] = fixed_point_16{temp_val_w1};
            
            // if (j == 0 && i == 0) {
            //     cout << "w1[0][0] = " << w1[0][0] << endl; // Debugging line
            //     printf("temp_val_w1 = %f\n", temp_val_w1); // Debugging line
            // }
            
            
            //w1[i][j] = fixed_point_16(0.5); // Default value converted to fixed-point
        }
    }

    // Load w2
    for (int i = 0; i < n_output; ++i) {
        for (int j = 0; j < n_hidden + 1; ++j) {
            // If reading from file:
            double temp_val_w2;
            if (!(file_w2 >> temp_val_w2)) {
                cerr << "Error: Could not read weight w2[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w2[i][j] = fixed_point_16{temp_val_w2};
            if (w2[i][j] == 0) {
                cerr << "Warning: Zero weight found at w2[" << i << "][" << j << "]." << endl;
            }
            //w2[i][j] = fixed_point_16(0.5); // Default value converted to fixed-point
        }
    }

     file_w1.close();
     file_w2.close();

    cout << "Weights loaded successfully." << endl;
    return true;
}

int main() {
    // Specify the filenames where the trained weights are stored
    string weights_file_w1 = "weights_w1_decFix.txt";
    string weights_file_w2 = "weights_w2_decFix.txt";

    // Load the trained weights
    if (!load_weights(weights_file_w1, weights_file_w2)) {
        return 1;
    }

    // Example inference
    array<fixed_point_16, n_features> input_sample = {fixed_point_16(0.0), fixed_point_16(0.0)};
    fixed_point_16 prediction = MLP_predict_single_sample(input_sample);

    // CNL types usually have operator<< overloaded for cout.
    // If not, or for specific float-like output, cast to double: static_cast<double>(value)
    cout << "Input: [" << input_sample[0] << ", " << input_sample[1] << "]" << endl;
    cout << "Prediction: [" << prediction << "]" << endl;

    return 0;
}