#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;

// //////////////////////////////////////////// //
//             MLP parameters for inference     //
// //////////////////////////////////////////// //
const int n_output = 1;         // Number of outputs
const int n_features = 2;         // Number of input features
const int n_hidden = 300;         // Number of neurons in the hidden layer

// Weights loaded from a file
array<array<float, n_features + 1>, n_hidden> w1;
array<array<float, n_hidden + 1>, n_output> w2;

// Forward propagation variables
array<float, n_features + 1> a0_infer;
array<float, n_hidden> rZ1_infer;
array<float, n_hidden> rA1_infer;
array<float, n_hidden + 1> a1_infer;
array<float, n_output> rZ2_infer;
array<float, n_output> y_pred_infer;

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

// Sigmoid activation function
float sigmoid_infer(float z) {
    return 1.0 / (1.0 + exp(-z));
}

// Forward pass for a single input
array<float, n_output> MLP_inference(const array<float, n_features>& x) {
    // Input layer with bias
    a0_infer[0] = 1.0; // Bias
    for (int i = 0; i < n_features; ++i) {
        a0_infer[i + 1] = x[i];
    }

    // Hidden layer pre-activation
    for (int i = 0; i < n_hidden; ++i) {
        rZ1_infer[i] = 0.0;
        for (int j = 0; j < n_features + 1; ++j) {
            rZ1_infer[i] += w1[i][j] * a0_infer[j];
        }
    }

    // Hidden layer activation
    for (int i = 0; i < n_hidden; ++i) {
        rA1_infer[i] = sigmoid_infer(rZ1_infer[i]);
    }

    // Hidden layer output with bias
    a1_infer[0] = 1.0; // Bias
    for (int i = 0; i < n_hidden; ++i) {
        a1_infer[i + 1] = rA1_infer[i];
    }

    // Output layer pre-activation
    for (int i = 0; i < n_output; ++i) {
        rZ2_infer[i] = 0.0;
        for (int j = 0; j < n_hidden + 1; ++j) {
            rZ2_infer[i] += w2[i][j] * a1_infer[j];
        }
    }

    // Output layer (linear activation for regression)
    for (int i = 0; i < n_output; ++i) {
        y_pred_infer[i] = rZ2_infer[i];
    }

    return y_pred_infer;
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
    array<float, n_features> input_sample = {1.0, -2.0};
    array<float, n_output> prediction = MLP_inference(input_sample);

    cout << "Input: [" << input_sample[0] << ", " << input_sample[1] << "]" << endl;
    cout << "Prediction: [" << prediction[0] << "]" << endl;

    // You can add more inference examples here, or read input from a file.

    return 0;
}