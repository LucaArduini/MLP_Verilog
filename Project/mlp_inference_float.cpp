#include <iostream>
#include <array>
#include <cmath>
#include <fstream>

using namespace std;

// //////////////////////////////////////////// //
//             MLP parameters for inference     //
// //////////////////////////////////////////// //
const int n_output = 1;             // Number of outputs
const int n_features = 2;           // Number of input features
const int n_hidden = 4;           // Number of neurons in the hidden layer

// Weights are assumed to be loaded from a file
array<array<float, n_features + 1>, n_hidden> w1;
array<array<float, n_hidden + 1>, n_output> w2;

// Forward propagation variables
array<float, n_features + 1> a0_infer;
array<float, n_hidden> rZ1_infer;
array<float, n_hidden> rA1_infer;
array<float, n_hidden + 1> a1_infer;
array<float, n_output> rZ2_infer;
array<float, n_output> y_pred_infer;

// Matrix multiplication function
void A_mult_B(const float* A, const float* B, float* C,
              int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            C[i * colB + j] = 0.0;
            for (int k = 0; k < colA; k++) {
                
                C[i * colB + j] += A[i * colA + k] * B[k * colB + j];
                
            }
        }
        
    }
}


// This function computes the ReLU function for a scalar, a vector or a matrix
void MLP_relu_inplace(const array<float, n_hidden> &z, array<float, n_hidden> &relu_out){
    for (int i = 0; i < n_hidden; ++i) {
        
        relu_out[i] = max(0.0f, z[i]); // ReLU(x) = max(0, x)
        
    }
}

// Forward pass for a single input using matrix operations and MLP_relu_inplace
array<float, n_output> MLP_inference(const array<float, n_features>& x) {
    // Input layer with bias
    a0_infer[0] = 1.0; // Bias
    for (int i = 0; i < n_features; ++i) {
        a0_infer[i + 1] = x[i];
    }

    // Hidden layer pre-activation: rZ1 = w1 * a0
    A_mult_B(w1[0].data(), a0_infer.data(), rZ1_infer.data(),
             n_hidden, n_features + 1, 1); // Treat a0 as a column vector

    // Hidden layer activation: rA1 = ReLU(rZ1)
    MLP_relu_inplace(rZ1_infer, rA1_infer);

    // Hidden layer output with bias
    a1_infer[0] = 1.0; // Bias
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
    float value;

    // Load w1
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_features + 1; ++j) {
            if (!(file_w1 >> value)) {
                cerr << "Error: Could not read weight w1[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w1[i][j] = value;
            
        }
    }

    // Load w2
    for (int i = 0; i < n_output; ++i) {
        for (int j = 0; j < n_hidden + 1; ++j) {
            if (!(file_w2 >> value)) {
                cerr << "Error: Could not read weight w2[" << i << "][" << j << "]." << endl;
                file_w1.close();
                file_w2.close();
                return false;
            }
            w2[i][j] = value;
        }
    }

    file_w1.close();
    file_w2.close();

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
    array<float, n_features> input_sample = {0.0, 0.0};
    array<float, n_output> prediction = MLP_inference(input_sample);

    cout << "Input: [" << input_sample[0] << ", " << input_sample[1] << "]" << endl;
    cout << "Prediction: [" << prediction[0] << "]" << endl;

    return 0;
}