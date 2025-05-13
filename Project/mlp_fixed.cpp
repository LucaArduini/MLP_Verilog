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

// Define the fixed-point type
using fixed_point_32 = cnl::scaled_integer<int32_t, cnl::power<-20>>;

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
void A_mult_B(const fixed_point_32* A, const fixed_point_32* B, fixed_point_32* C,
              int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            C[i * colB + j] = fixed_point_32(0.0); // Initialize with fixed-point zero
            for (int k = 0; k < colA; k++) {
                C[i * colB + j] += A[i * colA + k] * B[k * colB + j];
            }
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

    printf("rZ1_infer[0] = %f\n", rZ1_infer[0]); // Debugging line

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