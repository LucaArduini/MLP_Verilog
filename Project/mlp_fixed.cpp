#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits> // For numeric_limits

using namespace std;

// //////////////////////////////////////////// //
//             MLP parameters for inference     //
// //////////////////////////////////////////// //
const int n_output = 1;         // Number of outputs
const int n_features = 2;         // Number of input features
const int n_hidden = 300;         // Number of neurons in the hidden layer

// Defining the fixed-point type
typedef int fixed_point_32;
const int FRACTIONAL_BITS = 16;

// Helper function to convert float to fixed-point
fixed_point_32 float_to_fixed(float val) {
    return static_cast<fixed_point_32>(round(val * (1 << FRACTIONAL_BITS)));
}

// Helper function to convert fixed-point to float (for verification)
float fixed_to_float(fixed_point_32 val) {
    return static_cast<float>(val) / (1 << FRACTIONAL_BITS);
}

// Fixed-point weights assumed to be loaded from a file
array<array<fixed_point_32, n_features + 1>, n_hidden> w1_fixed;
array<array<fixed_point_32, n_hidden + 1>, n_output> w2_fixed;

// Fixed-point forward propagation variables
array<fixed_point_32, n_features + 1> a0_infer_fixed;
array<fixed_point_32, n_hidden> rZ1_infer_fixed;
array<fixed_point_32, n_hidden> rA1_infer_fixed;
array<fixed_point_32, n_hidden + 1> a1_infer_fixed;
array<fixed_point_32, n_output> rZ2_infer_fixed;
array<fixed_point_32, n_output> y_pred_infer_fixed;

// Fixed-point matrix multiplication function (A_mult_B for fixed-point)
void fixed_A_mult_B(const fixed_point_32* A, const fixed_point_32* B, fixed_point_32* C,
                    int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; ++i) {
        for (int j = 0; j < colB; ++j) {
            long long temp_sum = 0;
            for (int k = 0; k < colA; ++k) {
                temp_sum += static_cast<long long>(A[i * colA + k]) * B[k * colB + j];
            }
            C[i * colB + j] = static_cast<fixed_point_32>(round(temp_sum / (1 << FRACTIONAL_BITS)));
        }
    }
}

// Fixed-point sigmoid approximation using a lookup table
array<fixed_point_32, n_hidden> fixed_MLP_sigmoid(const array<fixed_point_32, n_hidden>& z) {
    array<fixed_point_32, n_hidden> sig;
    const int LUT_SIZE = 1024; // size for accuracy vs. memory
    static array<fixed_point_32, LUT_SIZE> sigmoid_lut;
    fixed_point_32 input_range = float_to_fixed(10.0); // range of input values for sigmoid - [-5.0, 5.0]

    // Initialize the LUT
    static bool lut_initialized = false; // Ensure LUT is initialized only once
    if (!lut_initialized) {
        for (int i = 0; i < LUT_SIZE; ++i) {
            float float_input = -5.0f + (static_cast<float>(i) / LUT_SIZE) * 10.0f; // Map LUT index to input range
            sigmoid_lut[i] = float_to_fixed(1.0 / (1.0 + exp(-float_input)));
        }
        lut_initialized = true;
    }

    for (int i = 0; i < n_hidden; ++i) {
        // Scale and map the fixed-point input to the LUT index
        long long index = static_cast<long long>(z[i] - float_to_fixed(-5.0)) * (LUT_SIZE - 1) / input_range;
        if (index < 0) index = 0;
        if (index >= LUT_SIZE) index = LUT_SIZE - 1;
        sig[i] = sigmoid_lut[index];
    }
    return sig;
}

// Fixed-point forward pass for a single input
array<fixed_point_32, n_output> fixed_MLP_inference(const array<float, n_features>& x) {
    // Input layer with bias (convert float input to fixed-point)
    a0_infer_fixed[0] = float_to_fixed(1.0); // Bias
    for (int i = 0; i < n_features; ++i) {
        a0_infer_fixed[i + 1] = float_to_fixed(x[i]);
    }

    // Hidden layer pre-activation: rZ1 = w1 * a0
    fixed_A_mult_B(reinterpret_cast<fixed_point_32*>(w1_fixed.data()), a0_infer_fixed.data(), rZ1_infer_fixed.data(),
                   n_hidden, n_features + 1, 1);

    // Hidden layer activation: rA1 = sigmoid(rZ1)
    rA1_infer_fixed = fixed_MLP_sigmoid(rZ1_infer_fixed);

    // Hidden layer output with bias
    a1_infer_fixed[0] = float_to_fixed(1.0); // Bias
    for (int i = 0; i < n_hidden; ++i) {
        a1_infer_fixed[i + 1] = rA1_infer_fixed[i];
    }

    // Output layer pre-activation: rZ2 = w2 * a1
    fixed_A_mult_B(reinterpret_cast<fixed_point_32*>(w2_fixed.data()), a1_infer_fixed.data(), rZ2_infer_fixed.data(),
                   n_output, n_hidden + 1, 1);

    // Output layer (linear activation for regression)
    for (int i = 0; i < n_output; ++i) {
        y_pred_infer_fixed[i] = rZ2_infer_fixed[i];
    }

    array<fixed_point_32, n_output> result = y_pred_infer_fixed;
    return result;
}

// Function to load fixed-point weights from a file
bool load_fixed_point_weights(const string& filename_w1, const string& filename_w2) {
    ifstream file_w1(filename_w1);
    ifstream file_w2(filename_w2);

    if (!file_w1.is_open() || !file_w2.is_open()) {
        cerr << "Error: Could not open weight files." << endl;
        return false;
    }

    string line;
    float float_value;

    // Load w1
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_features + 1; ++j) {
            if (!(file_w1 >> float_value)) {
                cerr << "Error: Could not read weight w1[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w1_fixed[i][j] = float_to_fixed(float_value);
        }
    }

    // Load w2
    for (int i = 0; i < n_output; ++i) {
        for (int j = 0; j < n_hidden + 1; ++j) {
            if (!(file_w2 >> float_value)) {
                cerr << "Error: Could not read weight w2[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w2_fixed[i][j] = float_to_fixed(float_value);
        }
    }

    file_w1.close();
    file_w2.close();
    cout << "Fixed-point weights loaded successfully (16.16 format)." << endl;
    return true;
}

int main() {
    // Specify the filenames where the trained floating-point weights are stored
    string weights_file_w1 = "weights_w1.txt";
    string weights_file_w2 = "weights_w2.txt";

    // Load the trained weights into fixed-point format
    if (!load_fixed_point_weights(weights_file_w1, weights_file_w2)) {
        return 1;
    }

    // Example inference
    array<float, n_features> input_sample = {1.0, -2.0};
    array<fixed_point_32, n_output> prediction_fixed = fixed_MLP_inference(input_sample);
    float prediction_float = fixed_to_float(prediction_fixed[0]);

    cout << "Input: [" << input_sample[0] << ", " << input_sample[1] << "]" << endl;
    cout << "Fixed-point Prediction: [" << prediction_fixed[0] << "]" << endl;
    cout << "Floating-point Equivalent Prediction: [" << prediction_float << "]" << endl;


    return 0;
}