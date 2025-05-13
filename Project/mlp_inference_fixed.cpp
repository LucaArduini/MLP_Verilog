#include <iostream>
#include <vector>
#include <array>
#include <cmath> // Still useful for std::exp if cnl::exp isn't found or for comparison
#include <fstream>
#include <sstream>

#include <cnl/all.h> // For cnl::scaled_integer

// Using namespaces for convenience in this .cpp file
using namespace std;
using namespace cnl;
const int fractional_bits = 16; // Number of fractional bits for fixed-point representation   

// Define the fixed-point type
using fixed_point_32 = cnl::scaled_integer<int32_t, cnl::power<-fractional_bits>>;

// //////////////////////////////////////////// //
//             MLP parameters for inference     //
// //////////////////////////////////////////// //
const int n_output = 1;         // Number of outputs
const int n_features = 2;         // Number of input features
const int n_hidden = 300;         // Number of neurons in the hidden layer

// Weights are assumed to be loaded from a file
array<array<fixed_point_32, n_features + 1>, n_hidden> w1;
array<array<fixed_point_32, n_hidden + 1>, n_output> w2;

// Forward propagation variables
array<fixed_point_32, n_features + 1> a0_infer;
array<fixed_point_32, n_hidden> rZ1_infer;
array<fixed_point_32, n_hidden> rA1_infer;
array<fixed_point_32, n_hidden + 1> a1_infer;
array<fixed_point_32, n_output> rZ2_infer;
array<fixed_point_32, n_output> y_pred_infer;

// Matrix multiplication function
void A_mult_B(const fixed_point_32* A_data, const fixed_point_32* B_data, fixed_point_32* C_data,
                          int rigA, int colA, int colB) {

    // 1. Get the underlying representation type of fixed_point_32
    //    Since fixed_point_32 is cnl::scaled_integer<int, cnl::power<-16>>,
    //    the Rep type is 'int'.
    using InputRep = int;

    // 2. Determine a wider representation type for the accumulator
    //    For InputRep = int, int64_t is typical.
    using AccumulatorRep = int64_t;

    // 3. Get the scaling information (the cnl::power<Exponent> tag)
    //    Since fixed_point_32 is cnl::scaled_integer<int, cnl::power<-16>>,
    //    the Scale tag is cnl::power<-16>.
    using InputScaleTag = cnl::power<-20>; // Adjusted for 32-bit representation

    // Define the accumulator type
    using Accumulator = cnl::scaled_integer<int64_t, cnl::power<-32>>;

    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            Accumulator temp_sum = Accumulator(0LL);

            for (int k = 0; k < colA; k++) {
                auto product = A_data[i * colA + k] * B_data[k * colB + j];
                temp_sum += product;
                if (i == 0 && j == 0) {
                    cout << "A_data[" << i << "][" << k << "] * B_data[" << k << "][" << j << "] = " 
                         << A_data[i * colA + k] << " * " << B_data[k * colB + j] 
                         << " = " << A_data[i * colA + k] * B_data[k * colB + j] << endl; // Debugging line
                }
            }

            C_data[i * colB + j] = static_cast<fixed_point_32>(temp_sum);
            if (i == 0 && j == 0) {
                cout << "C_data[" << i << "][" << j << "] = " << C_data[i * colB + j] << endl; // Debugging line
            }   
            
        }
        if (i == 0) {
            cout << "C_data[0] (cout): " << C_data[0] << endl; // Debugging line
        }
        
    }
}


// Sigmoid activation function using fixed-point numbers
array<fixed_point_32, n_hidden> MLP_sigmoid(const array<fixed_point_32, n_hidden> &z) {
    array<fixed_point_32, n_hidden> sig;
    for (int i = 0; i < n_hidden; ++i) {
        // Use cnl::exp for fixed-point types.
        // The expression will likely involve promotions to a floating-point type for exp,
        // then conversion back to fixed_point_32.
        sig[i] = fixed_point_32(1.0) / (fixed_point_32(1.0) + cnl::exp(-z[i]));
    }
    return sig;
}

// Forward pass for a single input using matrix operations and MLP_sigmoid
array<fixed_point_32, n_output> MLP_inference(const array<fixed_point_32, n_features>& x) {
    // Input layer with bias
    a0_infer[0] = fixed_point_32(1.0); // Bias
    for (int i = 0; i < n_features; ++i) {
        a0_infer[i + 1] = x[i];
    }

    // Hidden layer pre-activation: rZ1 = w1 * a0
    A_mult_B(w1[0].data(), a0_infer.data(), rZ1_infer.data(),
             n_hidden, n_features + 1, 1); // Treat a0 as a column vector

    cout << "rZ1_infer[0] (cout): " << rZ1_infer[0] << endl;
    
    // Hidden layer activation: rA1 = sigmoid(rZ1)
    rA1_infer = MLP_sigmoid(rZ1_infer);

    // Hidden layer output with bias
    a1_infer[0] = fixed_point_32(1.0); // Bias
    for (int i = 0; i < n_hidden; ++i) {
        a1_infer[i + 1] = rA1_infer[i];
    }

    // Output layer pre-activation: rZ2 = w2 * a1
    A_mult_B(w2[0].data(), a1_infer.data(), rZ2_infer.data(),
             n_output, n_hidden + 1, 1); // Treat a1 as a column vector

    // Output layer (linear activation for regression)
    for (int i = 0; i < n_output; ++i) {
        y_pred_infer[i] = rZ2_infer[i];
    }

    return y_pred_infer;
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
    fixed_point_32 value; // Changed to fixed-point

    // Load w1
    double maxerror = 0.0;
    double meanerror = 0.0;
    for (int i = 0; i < n_hidden; ++i) {
        
        for (int j = 0; j < n_features + 1; ++j) {
            //If reading from file:
            double temp_val_w1;
            if (!(file_w1 >> temp_val_w1)) {
                cerr << "Error: Could not read weight w1[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w1[i][j] = fixed_point_32(temp_val_w1);
            
            if (temp_val_w1 - w1[i][j] > maxerror) {
                maxerror = temp_val_w1 - w1[i][j];
            }
            meanerror += temp_val_w1 - w1[i][j];
            if (j == 0 && i == 0) {
                cout << "w1[0][0] = " << w1[0][0] << endl; // Debugging line
                printf("temp_val_w1 = %f\n", temp_val_w1); // Debugging line
            }
            
            
            //w1[i][j] = fixed_point_32(0.5); // Default value converted to fixed-point
        }
    }
    printf("maxerror = %f\n", maxerror); // Debugging line
    meanerror /= (n_hidden * (n_features + 1)); // Debugging line
    printf("meanerror = %f\n", meanerror); // Debugging line

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
            w2[i][j] = fixed_point_32(temp_val_w2);
            //w2[i][j] = fixed_point_32(0.5); // Default value converted to fixed-point
        }
    }

    // file_w1.close();
    // file_w2.close();

    cout << "Weights loaded successfully." << endl;
    return true;
}

int main() {
    // Specify the filenames where the trained weights are stored
    string weights_file_w1 = "weights_w1.txt";
    string weights_file_w2 = "weights_w2.txt";

    // Load the trained weights
    if (!load_weights(weights_file_w1, weights_file_w2)) {
        return 1;
    }

    // Example inference
    array<fixed_point_32, n_features> input_sample = {fixed_point_32(1.0), fixed_point_32(-2.0)};
    array<fixed_point_32, n_output> prediction = MLP_inference(input_sample);

    // CNL types usually have operator<< overloaded for cout.
    // If not, or for specific float-like output, cast to double: static_cast<double>(value)
    cout << "Input: [" << input_sample[0] << ", " << input_sample[1] << "]" << endl;
    cout << "Prediction: [" << prediction[0] << "]" << endl;

    return 0;
}