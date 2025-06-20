
#include <iostream>                 // For std::cout, std::cerr, std::endl (console output, file error handling)
#include <vector>                   // For std::vector (used in sinc2D_gen, MLP_MSE_cost)
#include <array>                    // For std::array (used for weight matrices, activations, data)
#include <cmath>                    // For std::pow, std::sqrt, std::exp (commented out), std::sin, std::acos
#include <algorithm>                // For std::shuffle, std::max, std::copy, std::min_element, std::max_element
#include <random>                   // For std::default_random_engine (shuffling)
#include <ctime>                    // For std::time (seeding for rand and random_engine)
#include <limits>                   // For std::numeric_limits (cost initialization)
#include <fstream>                  // For std::ofstream (writing weight files)
#include <iomanip>                  // For std::fixed, std::setprecision (decimal output formatting)
#include <numeric>                  // For std::accumulate, std::iota (cost calculation, shuffling)

#include "cnl/include/cnl/all.h"    // For cnl::scaled_integer and related functionalities (fixed_point_16)

using namespace std;
namespace impl = cnl::_impl;        // Namespace alias for CNL implementation details

bool load_dataset = false; // Set to true to load dataset from files



// //////////////////////////////////////////// //
//                 Dataset parameters           //
// //////////////////////////////////////////// //

const int num_train = 150*150;      // number of training pattern (ensure this is a square number for sinc2D_gen)
const int num_test = 2250;          // Number of test patterns



// //////////////////////////////////////////// //
//                 MLP parameters               //
// //////////////////////////////////////////// //

const int n_output = 1;             // Number of outputs (neurons in the output layer)
const int n_features = 2;           // Number of input features
const int n_hidden = 4;             // Number of neurons in the hidden layer
const int epochs = 500;              // Number of training epochs
float eta = 1e-5;                   // Learning rate
const int minibatches = 30;         // Number of mini-batches for training

vector<float> cost;                 // Vector to store the cost (MSE) at each training step (per mini-batch, cleared per run)

// Weight matrices for the MLP
// w1: weights for the layer between input and hidden layer (n_hidden x (n_features + 1))
// The +1 accounts for the bias term.
array<array<float, n_features+1>, n_hidden> w1 = {};
// w2: weights for the layer between hidden and output layer (n_output x (n_hidden + 1))
// The +1 accounts for the bias term.
array<array<float, n_hidden+1>, n_output> w2 = {};



// //////////////////////////////////////////// //
//          Fixed-Point parameters            //
// //////////////////////////////////////////// //

const int TOTAL_BITS = 16;                                      // Total number of bits for the fixed-point representation
const int FRACTIONAL_BITS = 8;                                  // Number of fractional bits
const int INTEGER_BITS = TOTAL_BITS - FRACTIONAL_BITS;          // Number of integer bits

const double SCALE_FACTOR = std::pow(2.0, FRACTIONAL_BITS);     // Scale factor for manual fixed-point conversion
const long long MIN_INT_VAL = -(1LL << (TOTAL_BITS - 1));       // Minimum representable integer value (e.g., -32768 for 16 bits)
const long long MAX_INT_VAL = (1LL << (TOTAL_BITS - 1)) - 1;    // Maximum representable integer value (e.g.,  32767 for 16 bits)

// Define the fixed-point type using CNL library
// int16_t: underlying integer type (16 bits)
// cnl::power<-FRACTIONAL_BITS>: specifies the position of the binary point (8 fractional bits)
using fixed_point_16 = cnl::scaled_integer<int16_t, cnl::power<-FRACTIONAL_BITS>>;

// Global declaration of variables used in the train step
// `elem` is the number of elements (samples) in each mini-batch
const int elem = (num_train + minibatches -1 )/minibatches;     // Inputs used in each minibatch (ceiling division)



// //////////////////////////////////////////// //
//      Global arrays for Forward Propagation   //
// //////////////////////////////////////////// //
// These arrays store intermediate values during the forward pass for a single mini-batch.
// Their size is determined by `elem` (mini-batch size).

array<array<float, n_features>, elem> x_input;                  // Mini-batch input data (elem x n_features)
array<array<float, elem>, n_features> rA0;                      // Transposed input mini-batch (n_features x elem), "reduced Activation 0"
array<array<float, elem>, n_features+1> a0;                     // Extended input mini-batch with bias ( (n_features+1) x elem), "Activation 0"
array<array<float, elem>, n_hidden> rZ1;                        // Pre-activation values for the hidden layer (n_hidden x elem), "reduced Z 1"
array<array<float, elem>, n_hidden> rA1;                        // Activation values for the hidden layer (n_hidden x elem), "reduced Activation 1"
array<array<float, elem>, n_hidden+1> a1;                       // Extended hidden layer activations with bias ( (n_hidden+1) x elem), "Activation 1"
array<array<float, elem>, n_output> rZ2;                        // Pre-activation values for the output layer (n_output x elem), "reduced Z 2"
array<array<float, elem>, n_output> rA2;                        // Activation values for the output layer (n_output x elem), "reduced Activation 2" (also network output)



// //////////////////////////////////////////// //
//    Global arrays for Backpropagation         //
// //////////////////////////////////////////// //
// These arrays store intermediate values during the backpropagation pass for a single mini-batch.

array<array<float, elem>, n_output> dL_dZ2;                     // Gradient of Loss w.r.t. Z2 (pre-activation of output layer)
array<array<float, n_hidden+1>, n_output> dL_dW2;               // Gradient of Loss w.r.t. W2 (weights of output layer)
array<array<float, elem>, n_hidden+1> dL_dA1;                   // Gradient of Loss w.r.t. A1 (activation of hidden layer, including bias)
array<array<float, elem>, n_hidden> activation_prime_of_rZ1;    // Derivative of activation function at rZ1 (hidden layer pre-activation)
array<array<float, elem>, n_hidden> dL_drZ1;                    // Gradient of Loss w.r.t. rZ1 (pre-activation of hidden layer, excluding bias part)
array<array<float, n_features+1>, n_hidden> dL_dW1;             // Gradient of Loss w.r.t. W1 (weights of hidden layer)
array<array<float, n_features+1>, n_hidden> delta_W1_unscaled;  // Unscaled weight updates for W1
array<array<float, n_hidden+1>, n_output> delta_W2_unscaled;    // Unscaled weight updates for W2



// //////////////////////////////////////////// //
//                 Utilities functions          //
// //////////////////////////////////////////// //

/**
 * @brief Matrix multiplication: C = A * B
 * @param A Pointer to the first element of matrix A.
 * @param B Pointer to the first element of matrix B.
 * @param C Pointer to the first element of the result matrix C.
 * @param rigA Number of rows in A.
 * @param colA Number of columns in A (must be equal to rows in B).
 * @param colB Number of columns in B.
 */
void A_mult_B(const float* A, const float* B, float* C,
              int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            C[i * colB + j] = 0.0; // Initialize element C[i][j]
            for (int k = 0; k < colA; k++) {
                C[i*colB+j] += A[i*colA+k] * B[k*colB+j];
            }
        }
    }
}

/**
 * @brief Matrix multiplication: C = A * B^T (B transpose)
 * @param A Pointer to the first element of matrix A.
 * @param B Pointer to the first element of matrix B.
 * @param C Pointer to the first element of the result matrix C.
 * @param rigA Number of rows in A.
 * @param colA Number of columns in A (must be equal to columns in B for B^T).
 * @param rigB Number of rows in B (becomes columns in B^T).
 */
void A_mult_B_T(const float* A, const float* B, float* C,
                   int rigA, int colA, int rigB) { // rigB is effectively col(B^T)
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < rigB; j++) { // Iterate up to rigB, which is columns of B^T
            C[i * rigB + j] = 0.0;
            for (int k = 0; k < colA; k++) {
                // B[j*colA+k] accesses B as if it's B[j][k] (row j, col k of B)
                // This corresponds to element B^T[k][j]
                C[i*rigB+j] += A[i*colA+k] * B[j*colA+k]; // B is indexed like B[j][k] (row j, col k)
            }
        }
    }
}

/**
 * @brief Matrix multiplication: C = A^T * B (A transpose)
 * @param A Pointer to the first element of matrix A.
 * @param B Pointer to the first element of matrix B.
 * @param C Pointer to the first element of the result matrix C.
 * @param rigA Number of rows in A.
 * @param colA Number of columns in A.
 * @param colB Number of columns in B.
 */
void A_T_mult_B(const float* A, const float* B, float* C,
                int rigA, int colA, int colB) { // Result C will be colA x colB
    for (int i = 0; i < colA; i++) { // Iterate through columns of A (rows of A^T)
        for (int j = 0; j < colB; j++) { // Iterate through columns of B
            C[i * colB + j] = 0.0;
            for (int k = 0; k < rigA; k++) { // Iterate through rows of A (columns of A^T)
                // A[k*colA+i] accesses A as A[k][i] (row k, col i of A)
                // This corresponds to element A^T[i][k]
                C[i * colB + j] += A[k * colA + i] * B[k * colB + j]; // B is indexed B[k][j]
            }
        }
    }
}

/**
 * @brief Element-wise multiplication of two matrices: C[i][j] = A[i][j] * B[i][j]
 * @param A Pointer to the first element of matrix A.
 * @param B Pointer to the first element of matrix B.
 * @param C Pointer to the first element of the result matrix C.
 * @param rig Number of rows in A, B, and C.
 * @param col Number of columns in A, B, and C.
 */
void elem_mult_elem(const float* A, const float* B, float* C, int rig, int col) {
    for (int i = 0; i < rig; ++i) {
        for (int j = 0; j < col; ++j) {
            C[i*col+j] = A[i*col+j] * B[i*col+j];
        }
    }
}

/**
 * @brief Saves the weights of the first layer (w1) to two files:
 *        - weights_w1_dec.txt: Decimal representation.
 *        - weights_w1_bin.txt: Binary representation (fixed-point).
 */
void save_weights_w1() {
    ofstream outFile_dec("weights_w1_dec.txt");
    ofstream outFile_bin("weights_w1_bin.txt");

    if (!outFile_bin || !outFile_dec) {
        cerr << "Error opening file for writing w1" << endl;
        return;
    }

    const unsigned char* byte_ptr;  // Pointer to inspect bytes of the fixed-point number

    // Iterate through the weight matrix w1
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_features + 1; ++j) {

            // Write the decimal representation of the weight
            outFile_dec << fixed << setprecision(8) << w1[i][j] << (j == n_features ? "" : " ");

            // Convert the float to fixed-point representation using cnl::scaled_integer
            fixed_point_16 fixed_val(w1[i][j]);
            // Get a byte pointer to the fixed-point value to read its binary representation
            byte_ptr = static_cast<const unsigned char*>(static_cast<const void*>(&fixed_val));
        
            // Write the binary representation of the fixed-point number (MSB first)
            for (int byte_idx = sizeof(fixed_val) - 1; byte_idx >= 0; --byte_idx) {
                for (int bit_pos = CHAR_BIT - 1; bit_pos >= 0; --bit_pos) {
                    outFile_bin << ((byte_ptr[byte_idx] >> bit_pos) & 1);
                }
            }

            // Add a space between weights in the file, except for the last one in a row
            outFile_bin << (j == n_features ? "" : " ");

        }
        outFile_dec << endl;
        outFile_bin << endl;
    }

    outFile_dec.close();
    outFile_bin.close();

    // Check for errors during file writing
    if (outFile_dec.fail() || outFile_bin.fail())
        std::cerr << "Error during writing to w1 files." << std::endl;
    else
        std::cout << "Weights w1 saved." << endl;
}

/**
 * @brief Saves the weights of the second layer (w2) to two files:
 *        - weights_w2_dec.txt: Decimal representation.
 *        - weights_w2_bin.txt: Binary representation (fixed-point).
 */
void save_weights_w2() {
    ofstream outFile_dec("weights_w2_dec.txt");
    ofstream outFile_bin("weights_w2_bin.txt");

    if (!outFile_bin || !outFile_dec) {
        cerr << "Error opening file for writing w2" << endl;
        return;
    }

    const unsigned char* byte_ptr;  // Pointer to inspect bytes of the fixed-point number

    // Iterate through the weight matrix w2
    for (int i = 0; i < n_output; ++i) {
        for (int j = 0; j < n_hidden + 1; ++j) {

            // Write the decimal representation of the weight
            outFile_dec << fixed << setprecision(8) << w2[i][j] << (j == n_hidden ? "" : " ");

            // Convert the float to fixed-point representation using cnl::scaled_integer
            fixed_point_16 fixed_val(w2[i][j]);
            // Get a byte pointer to the fixed-point value
            byte_ptr = static_cast<const unsigned char*>(static_cast<const void*>(&fixed_val));
        
            // Write the binary representation of the fixed-point number (MSB first)
            for (int byte_idx = sizeof(fixed_val) - 1; byte_idx >= 0; --byte_idx) {
                for (int bit_pos = CHAR_BIT - 1; bit_pos >= 0; --bit_pos) {
                    outFile_bin << ((byte_ptr[byte_idx] >> bit_pos) & 1);
                }
            }

            // Add a space between weights in the file, except for the last one in a row
            outFile_bin << (j == n_hidden ? "" : " ");

        }
        outFile_dec << endl;
        outFile_bin << endl;
    }

    outFile_dec.close();
    outFile_bin.close();

    // Check for errors during file writing
    if (outFile_dec.fail() || outFile_bin.fail())
        std::cerr << "Error during writing to w2 files." << std::endl;
    else
        std::cout << "Weights w2 saved." << endl;
}



/**
 * @brief Generates a 2D sinc dataset.
 *        The function creates a grid of points (x1, x2) and computes y = 10 * sinc(x1) * sinc(x2).
 * @param x Pointer to the array to store input features (x1, x2 pairs).
 *          The array is filled row-major, i.e., [x1_0, x2_0, x1_1, x2_1, ...].
 * @param y Pointer to the array to store output values.
 * @param num_patterns Total number of data patterns to generate. Must be a perfect square.
 */
void sinc2D_gen(float* x_ptr, float* y_ptr, int num_patterns){ // Changed x, y to x_ptr, y_ptr for clarity
    int num_points = sqrt(num_patterns);        // Number of points along each axis (x1, x2)

    // Generate linearly spaced points for x1 axis
    vector<float> x1_coords(num_points);
    float start_x1 = -5.0;
    float end_x1 = 5.0;
    float step_x1 = (num_points == 1) ? 0.0f : (end_x1 - start_x1) / (num_points - 1);
    for (int i = 0; i < num_points; ++i){
        x1_coords[i] = start_x1 + i * step_x1;
    }


    // Generate linearly spaced points for x2 axis
    vector<float> x2_coords(num_points);
    float start_x2 = -5.0;
    float end_x2 = 5.0;
    float step_x2 = (num_points == 1) ? 0.0f : (end_x2 - start_x2) / (num_points - 1);
    for (int i = 0; i < num_points; ++i){
        x2_coords[i] = start_x2 + i * step_x2;
    }


    // Create a meshgrid (XX1, XX2) from x1_coords and x2_coords
    vector<vector<float>> XX1(num_points, vector<float>(num_points));
    vector<vector<float>> XX2(num_points, vector<float>(num_points));
    for (int i = 0; i < num_points; ++i){           // Corresponds to x2_coords index
        for (int j = 0; j < num_points; ++j){       // Corresponds to x1_coords index
            XX1[i][j] = x1_coords[j];               // x1 varies along columns
            XX2[i][j] = x2_coords[i];               // x2 varies along rows
        }
    }


    // Compute sinc2D: YY[i][j] = 10 * sinc(XX1[i][j]) * sinc(XX2[i][j])
    vector<vector<float>> YY(num_points, vector<float>(num_points));
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            float val_x1 = XX1[i][j];
            float val_x2 = XX2[i][j];
            float sinc_x1 = (val_x1 == 0.0f) ? 1.0f : sin(val_x1) / val_x1;
            float sinc_x2 = (val_x2 == 0.0f) ? 1.0f : sin(val_x2) / val_x2;
            YY[i][j] = 10.0f * sinc_x1 * sinc_x2;
        }
    }


    // Initialize output arrays x and y
    for (int col_idx = 0; col_idx < num_points; ++col_idx) {         // Outer loop iterates over 'j' in original
        for (int row_idx = 0; row_idx < num_points; ++row_idx) {     // Inner loop iterates over 'i' in original
            int pattern_idx = col_idx * num_points + row_idx;
            x_ptr[pattern_idx * n_features + 0] = XX1[row_idx][col_idx]; // XX1[i][j] from original indexing
            x_ptr[pattern_idx * n_features + 1] = XX2[row_idx][col_idx]; // XX2[i][j] from original indexing
            y_ptr[pattern_idx]                  = YY[row_idx][col_idx];  // YY[i][j] from original indexing
        }
    }
}

//Read the dataset from files
void read_dataset_train(array<array<float, n_features>, num_train> &x_train_data, array<float, num_train> &y_train_data) {
    ifstream x_file("sinc2D_x_train.txt");
    ifstream y_file("sinc2D_y_train.txt");

    if (!x_file.is_open() || !y_file.is_open()) {
        cerr << "Error opening training dataset files." << endl;
        // Consider exiting or throwing an exception if files are critical
        return;
    }

    for (int i = 0; i < num_train; ++i) {
        for (int j = 0; j < n_features; ++j) {
            x_file >> x_train_data[i][j];
        }
        y_file >> y_train_data[i];
    }

    x_file.close();
    y_file.close();
}

void read_dataset_test(array<array<float, n_features>, num_test> &x_test_data, array<float, num_test> &y_test_data) {
    ifstream x_file("sinc2D_x_test.txt");
    ifstream y_file("sinc2D_y_test.txt");

    if (!x_file.is_open() || !y_file.is_open()) {
        cerr << "Error opening test dataset files." << endl;
        // Consider exiting or throwing an exception
        return;
    }

    for (int i = 0; i < num_test; ++i) {
        for (int j = 0; j < n_features; ++j) {
            x_file >> x_test_data[i][j];
        }
        y_file >> y_test_data[i];
    }

    x_file.close();
    y_file.close();
}



// ////////////////////////////////////////////////////////////////////////// //
//                  ACTIVATION FUNCTIONS AND THEIR GRADIENTS                  //  
// ////////////////////////////////////////////////////////////////////////// //

/**
 * @brief Computes the ReLU (Rectified Linear Unit) activation function in-place.
 *        ReLU(x) = max(0, x).
 * @param z Input matrix (pre-activations), n_hidden x elem.
 * @param relu_out Output matrix (activations), n_hidden x elem.
 */
void MLP_relu_inplace(const array<array<float, elem>, n_hidden> &z, array<array<float, elem>, n_hidden> &relu_out){
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            relu_out[i][j] = max(0.0f, z[i][j]); // ReLU(x) = max(0, x)
        }
    }
}


/**
 * @brief Computes the gradient of the ReLU activation function in-place.
 *        ReLU'(x) = 1 if x > 0, else 0.
 * @param Z Input matrix (pre-activations where ReLU was applied), n_hidden x elem.
 * @param reluGrad_out Output matrix (gradients), n_hidden x elem.
 */
void MLP_relu_gradient_inplace(const array<array<float, elem>, n_hidden> &Z, array<array<float, elem>, n_hidden> &reluGrad_out) {
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            reluGrad_out[i][j] = (Z[i][j] > 0.0f) ? 1.0f : 0.0f; // Derivative: 1 if Z > 0, else 0
        }
    }
}


/**
 * @brief Performs the forward pass of the MLP.
 *        Calculates activations from input to output layer for a mini-batch.
 *        Output layer uses a linear activation function (identity).
 *        Hidden layer uses ReLU activation.
 */
void MLP_MSELIN_forward(){
    // Step 1: Prepare rA0 (reduced Activation 0)
    // rA0 is the transpose of the input mini-batch x_input.
    // x_input is (elem x n_features), rA0 becomes (n_features x elem).
    for (int i = 0; i < elem; ++i) {        // Iterate through samples in mini-batch
        for (int j = 0; j < n_features; ++j) { // Iterate through features
            rA0[j][i] = x_input[i][j];
        }
    }

    // Step 2: Prepare a0 (Activation 0) by adding bias term to rA0.
    // a0 is ((n_features+1) x elem). The first row is all ones (bias).
    for (int i = 0; i < n_features+1; ++i) { // Iterate rows of a0
        for (int j = 0; j < elem; ++j) {     // Iterate columns (samples) of a0
            a0[i][j] = (i == 0) ? 1.0f : rA0[i-1][j]; // First row is bias (1), others are from rA0
        }
    }

    // Step 3: Calculate rZ1 (pre-activation for hidden layer).
    // rZ1 = W1 * a0.
    // W1 is (n_hidden x (n_features+1)), a0 is ((n_features+1) x elem).
    // rZ1 will be (n_hidden x elem).
    A_mult_B(w1[0].data(), a0[0].data(), rZ1[0].data(), n_hidden, n_features+1, elem);

    // Step 4: Calculate rA1 (activation for hidden layer) using ReLU.
    // rA1 = ReLU(rZ1).
    // rA1 is (n_hidden x elem).
    MLP_relu_inplace(rZ1, rA1);

    // Step 5: Prepare a1 (Activation 1) by adding bias term to rA1.
    // a1 is ((n_hidden+1) x elem). The first row is all ones (bias).
    for (int i = 0; i < n_hidden+1; ++i) { // Iterate rows of a1
        for (int j = 0; j < elem; ++j) {     // Iterate columns (samples) of a1
            a1[i][j] = (i == 0) ? 1.0f : rA1[i-1][j]; // First row is bias (1), others are from rA1
        }
    }

    // Step 6: Calculate rZ2 (pre-activation for output layer).
    // rZ2 = W2 * a1.
    // W2 is (n_output x (n_hidden+1)), a1 is ((n_hidden+1) x elem).
    // rZ2 will be (n_output x elem).
    A_mult_B(w2[0].data(), a1[0].data(), rZ2[0].data(), n_output, n_hidden+1, elem);

    // Step 7: Calculate rA2 (activation for output layer).
    // For MSE with linear output, rA2 = rZ2 (identity activation).
    // rA2 is (n_output x elem). This is the network's prediction.
    rA2 = rZ2;
}



/**
 * @brief Initializes the weight matrices W1 and W2 with random values.
 *        Weights are drawn from a uniform distribution U[-1, 1].
 *        The weights are initialized in a way that, if flattened, corresponds to column-major order.
 */
void MLP_initialize_weights(){
    // Initialize W1: (n_hidden x (n_features+1))
    array<float, n_hidden*(n_features+1)> w1_temp; // Temporary flat array for W1
    for(int i=0; i<n_hidden*(n_features+1); ++i){
        w1_temp[i] = 2.0f * (static_cast<float>(rand())/RAND_MAX) - 1.0f; // Random value in [-1, 1]
    }

    // Reshape w1_temp into w1 (column-major like filling)
    int index = 0;
    for (int j = 0; j < (n_features+1); ++j) { // Iterate through columns of W1
        for (int i = 0; i < n_hidden; ++i) {   // Iterate through rows of W1
            w1[i][j] = w1_temp[index++];
        }
    }

    // Initialize W2: (n_output x (n_hidden+1))
    array<float, n_output*(n_hidden+1)> w2_temp; // Temporary flat array for W2
    for(int i=0; i<n_output*(n_hidden+1); ++i){
        w2_temp[i] = 2.0f * (static_cast<float>(rand())/RAND_MAX) - 1.0f; // Random value in [-1, 1]
    }

    // Reshape w2_temp into w2 (column-major like filling)
    index = 0;
    for (int j = 0; j < (n_hidden+1); ++j) { // Iterate through columns of W2
        for (int i = 0; i < n_output; ++i) { // Iterate through rows of W2
            w2[i][j] = w2_temp[index++];
        }
    }
}


/**
 * @brief Calculates the Mean Squared Error (MSE) cost.
 *        MSE = (1 / (2 * N)) * sum((y_true - y_pred)^2)
 * @param y_true Array of true target values for the current mini-batch (1 x elem).
 * @return The calculated MSE cost.
 */
float MLP_MSE_cost(const array<float, elem> &y_true) { // y_true is effectively a row vector
    vector<float> diff(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        // rA2[0][i] is the prediction for the i-th sample in the mini-batch
        // (assuming n_output is 1, so rA2 has one row)
        diff[i] = (y_true[i] - rA2[0][i]);
        diff[i] *= diff[i]; // Square the difference
    }

    // Sum of squared differences
    float sum_sq_diff = accumulate(diff.begin(), diff.end(), 0.0f);
    // Calculate MSE
    float cost_val = sum_sq_diff / (2.0f * y_true.size());
    return cost_val;
}


/**
 * @brief Performs the backpropagation step to compute gradients of the loss
 *        with respect to weights W1 and W2.
 *        Assumes MSE loss and linear output activation, ReLU hidden activation.
 * @param y_true Array of true target values for the current mini-batch (1 x elem).
 */
void MLP_MSELIN_backprop(const array<float, elem> &y_true){
    // Matrix dimensions for reference (B = elem, D = n_features, H = n_hidden):
    // rA2: (n_output x B)  -- Network predictions
    // a1:  ((H+1) x B)     -- Hidden layer activations (with bias)
    // a0:  ((D+1) x B)     -- Input layer activations (with bias)
    // rZ1: (H x B)         -- Hidden layer pre-activations
    // y_true: (1 x B)       -- True labels (passed as 1D array, conceptually row vector)
    // W1:  (H x (D+1))
    // W2:  (n_output x (H+1))


    // Step 1: Compute dL_dZ2 (gradient of Loss w.r.t. Z2, pre-activation of output layer)
    // For MSE loss L = 1/2N * sum( (rA2 - y_true)^2 ) and linear output rA2 = Z2,
    // dL/dZ2 = dL/drA2 * drA2/dZ2 = (rA2 - y_true) * 1 = rA2 - y_true (element-wise for each sample)
    // dL_dZ2 dimensions: (n_output x elem)
    // NB: rA2 coincides with y_pred
    // NB: dL_dZe could be called "grad2", the gradient on the output layer with respect the pre-activation Z2
    for(int i = 0; i<n_output; i++){
        for(int j = 0; j < elem; ++j) {
            dL_dZ2[i][j] = rA2[i][j] - y_true[j];
        }
    }


    // Step 2: Compute dL_dW2 (gradient of Loss w.r.t. W2)
    // dL_dW2 = dL_dZ2 * a1^T
    // dL_dZ2 is (n_output x elem), a1 is ((n_hidden+1) x elem) -> a1^T is (elem x (n_hidden+1))
    // dL_dW2 dimensions: (n_output x (n_hidden+1)), same as W2
    // NB: dL_dW2 could be called "delta_W2_unscaled", because it is of the same size of W2 and stores the unscaled variation
    A_mult_B_T(dL_dZ2[0].data(), a1[0].data(), dL_dW2[0].data(), n_output, elem, n_hidden+1);


    // Step 3: Compute dL_dA1 (gradient of Loss w.r.t. A1, activation of hidden layer including bias row)
    // dL_dA1 = W2^T * dL_dZ2
    // W2 is (n_output x (n_hidden+1)) -> W2^T is ((n_hidden+1) x n_output)
    // dL_dZ2 is (n_output x elem)
    // dL_dA1 dimensions: ((n_hidden+1) x elem)
    A_T_mult_B(w2[0].data(), dL_dZ2[0].data(), dL_dA1[0].data(), n_output, n_hidden+1, elem);


    // Step 4: Compute dL_drZ1 (gradient of Loss w.r.t. rZ1, pre-activation of hidden layer, excluding bias contribution)
    // dL_drZ1 = dL_drA1 * ReLU'(rZ1)
    // dL_drA1 is dL_dA1 but without the first row (which corresponds to bias, and derivative w.r.t. bias is not needed here for dZ1)
    // Effectively, dL_drA1 corresponds to dL_dA1[1...H_end].
    // ReLU'(rZ1) is the derivative of the ReLU function applied to rZ1.
    // dL_drZ1 dimensions: (n_hidden x elem)
    
    // Calculate derivative of ReLU activation for rZ1
    // activation_prime_of_rZ1 dimensions: (n_hidden x elem)
    MLP_relu_gradient_inplace(rZ1, activation_prime_of_rZ1);

    // Element-wise multiplication: dL_drZ1 = dL_dA1 (rows 1 to end) * activation_prime_of_rZ1
    // dL_dA1[0].data() points to the start of the ( (n_hidden+1) x elem ) matrix.
    // dL_dA1[1].data() would point to the start of the second row of dL_dA1.
    // Here, we are passing dL_dA1[1] (which is conceptually the start of the non-bias part of dL_dA1)
    // and treating it as a (n_hidden x elem) matrix.
    elem_mult_elem(dL_dA1[1].data(), activation_prime_of_rZ1[0].data(),dL_drZ1[0].data(),  n_hidden, elem);



    // Step 5: Compute dL_dW1 (gradient of Loss w.r.t. W1)
    // dL_dW1 = dL_drZ1 * a0^T
    // dL_drZ1 is (n_hidden x elem), a0 is ((n_features+1) x elem) -> a0^T is (elem x (n_features+1))
    // dL_dW1 dimensions: (n_hidden x (n_features+1)), same as W1
    // NB: dL_dW1 could be called "delta_W1_unscaled", because it is of the same size of W2 and stores the unscaled variation of W1
    A_mult_B_T(dL_drZ1[0].data(), a0[0].data(), dL_dW1[0].data(), n_hidden, elem, n_features+1);



    // Step 6: Store unscaled gradients (no regularization applied in this version)
    // These will be scaled by the learning rate later.
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_features+1; ++j) {
            delta_W1_unscaled[i][j] = dL_dW1[i][j];
        }
    }

    for (int j = 0; j < n_output; ++j) {
        for (int i = 0; i < n_hidden+1; ++i) {
            delta_W2_unscaled[j][i] = dL_dW2[j][i];
        }
    }


    /* -----------------------------------------------------------------------------
    NB: grad2 is the gradient at the hidden layer.
    It is a column vector in the case of a single pattern
    (minibatch equal to the training set site) or a matrix,
    to be imagined, in the latter case, a matrix of columns,
    the gradients of each input pattern in the minibatch.

    NB: grad1 is the gradient at the hidden layer (derivative
    of the loss with respect Z1, the pre-activation at the hidden layer).
    It is a column vector in the case of a single pattern
    (minibatch equal to the training set site) or a matrix,
    to be imagined, in the latter case, a matrix of columns,
    the gradients of each input pattern in the minibatch.
    ----------------------------------------------------------------------------- */
}

/**
 * @brief Trains the MLP using the provided training data.
 *        Implements mini-batch gradient descent.
 * @param x Training input data (num_train x n_features).
 * @param y Training target data (num_train x 1).
 */
void MLP_MSELIN_train(const array<array<float, n_features>, num_train> &x, const array<float, num_train> &y){
    // Initialize weights W1 and W2
    MLP_initialize_weights();

    // Clear cost history for this training run
    cost.clear();


    // Main training loop over epochs
    for(int e=1; e<=epochs; e++) {

        // Prepare mini-batch indices
        // I[m][k] will be the k-th global index for the m-th minibatch.
        // This creates `minibatches` number of index arrays, each of size `elem`.
        array<array<int, elem>, minibatches> I; // Should be sized correctly based on num_train and elem
                                                // If num_train is not a multiple of minibatches*elem, last minibatch might be smaller.
                                                // The current I calculation populates it as if all minibatches are full.
                                                // This is okay as long as 'elem' is calculated as ceil(num_train/minibatches)
                                                // and subsequent loops over 'idx' only process valid indices up to num_train.
                                                // The current loop for I:
                                                // for (int i = 0; i < num_train; ++i) { I[i % minibatches][i / minibatches] = i; } is correct.


        for (int i = 0; i < num_train; ++i) {
            int row = i % minibatches; // minibatch index
            int col = i / minibatches; // index within that minibatch's conceptual list
            if (col < elem) { // Ensure we don't write out of bounds for I's second dimension
                 I[row][col] = i;
            }
        }
        // If num_train is not a multiple of minibatches, some later entries in I might not be filled by the above.
        // And if elem * minibatches > num_train, then some idx arrays might be too long / contain uninitialized values.
        // Let's assume elem * minibatches >= num_train, and each minibatch 'm' will process its assigned 'idx'
        // making sure not to go past num_train effectively.
        // The loops using 'idx' later correctly use 'elem' as their size.
        // The x_input loading and y_index loading are also based on 'elem'.
        // If the last minibatch is smaller than 'elem', those loops still iterate 'elem' times.
        // This is okay if MLP_MSELIN_forward and backprop handle it (e.g. by processing up to actual_batch_size)
        // or if x_input/y_index are padded for the unused slots.
        // Given that elem = ceil(num_train / minibatches), all 'elem' slots will be used by some data point or padded.
        // The current code assumes each minibatch effectively processes 'elem' samples. If the last actual
        // minibatch is smaller, the extra slots in x_input are from earlier in x or uninitialized.
        // This should be handled like in MLP_MSELIN_predict, by considering actual batch size.
        // For simplicity now, we assume fixed mini-batch size `elem`. The last batch might contain some wrap-around if not careful with indices,
        // but the current indexing I[row][col] = i; seems to distribute indices correctly.

        // Loop over mini-batches
        for(int m_idx=0; m_idx<minibatches; ++m_idx){ // Iterate 0 to minibatches-1
            // array<int, elem> idx = I[m_idx]; // Get indices for the current mini-batch. Now I is num_train long.
            // The I structure is minibatches x elem.
            
            int current_batch_actual_size = 0;
            array<int, elem> idx; // Indices for the current minibatch
            idx.fill(-1); // Initialize with invalid index

            // Collect indices for the current minibatch m_idx
            // Each minibatch should get 'elem' samples if possible.
            // Example: num_train=100, minibatches=10, elem=10. I[0..9][0..9]
            // Example: num_train=105, minibatches=10, elem=11. I[0..9][0..10]
            // Minibatch m_idx gets samples: m_idx*elem to m_idx*elem + elem -1 (roughly)
            // The original I formation: I[i%minibatches][i/minibatches] = i
            // This means I[m] contains elements: m, m+minibatches, m+2*minibatches ...
            // This is a strided way of forming minibatches. It's valid.
            
            // For minibatch 'm_idx', its elements are I[m_idx][0], I[m_idx][1], ..., I[m_idx][elem-1]
            // We need to check if I[m_idx][k] is a valid index (i.e. < num_train)
            current_batch_actual_size = 0;
            for(int k=0; k<elem; ++k) {
                int original_data_idx = I[m_idx][k]; // This uses the I array as defined (strided)
                // Need to ensure original_data_idx is valid if num_train is not a multiple of (elem*minibatches)
                // The way 'elem' is calculated (ceiling) and 'I' is populated should mostly handle this.
                // If I[m_idx][k] was populated by an i >= num_train, that's an issue.
                // The loop `for (int i = 0; i < num_train; ++i)` ensures `I[row][col]` gets `i < num_train`.
                // Some slots in `I` might remain unassigned if `elem * minibatches > num_train` heavily.
                // Let's assume I is correctly populated with valid indices from 0 to num_train-1.

                if (original_data_idx < num_train && original_data_idx >= 0) { // Check if index is valid (actually, I should be pre-filled with valid indices)
                                                                        // The check `col < elem` in I population ensures this.
                    idx[k] = original_data_idx; // This is the original index from the shuffled full dataset
                                                // So x[idx[k]] is correct.
                    // Copy data for x_input
                    // The 'x' passed to MLP_MSELIN_train is already shuffled.
                    copy(x[idx[k]].begin(), x[idx[k]].end(), x_input[k].begin()); // x_input is elem x n_features
                                                                                // So x_input[k] is the k-th sample in minibatch
                    current_batch_actual_size++;
                } else {
                    // This case (invalid index from I) should ideally not happen with correct I population.
                    // If it does, pad x_input for this slot.
                    for(int f=0; f<n_features; ++f) x_input[k][f] = 0.0f; // Pad
                }
            }
            // If current_batch_actual_size < elem, the remaining x_input slots (from current_batch_actual_size to elem-1)
            // need to be padded if not already handled by the else clause above.
            // The current loop structure for x_input loads data sample by sample using k as row index for x_input.
            // This is different from the original code's x_input loading.
            // Original: for(int i=0; i<elem; i++) { copy(x[idx[i]].begin(), x[idx[i]].end(), x_input[i].begin()); }
            // This original way is simpler if idx directly contains the 'elem' indices for the current batch.
            // Let's revert to the simpler original way of populating x_input, assuming idx for the batch is correctly formed.
            
            // Get indices for the current mini-batch m_idx
            // The I array is minibatches x elem. I[m_idx] is the array of 'elem' indices for this minibatch.
            // These indices are into the (already shuffled) x and y.
            const auto& current_minibatch_indices = I[m_idx];

            for(int k=0; k<elem; ++k) {
                int data_idx_in_shuffled_set = current_minibatch_indices[k];
                // Check if this index is valid (it should be if I is populated correctly from 0 to num_train-1 indices)
                // If num_train is not a multiple of elem, the last few indices in the last minibatch might be invalid or repeated.
                // The original code `I[row][col] = i;` ensures that I[m_idx][k] refers to some `i < num_train`.
                // So, `x[data_idx_in_shuffled_set]` should be safe.
                if (data_idx_in_shuffled_set < num_train) { // This check should be redundant if I is perfect
                    copy(x[data_idx_in_shuffled_set].begin(), x[data_idx_in_shuffled_set].end(), x_input[k].begin());
                } else {
                    // This would be an error in minibatch formation logic. For safety, pad.
                    for(int f=0; f<n_features; ++f) x_input[k][f] = 0.0f;
                }
            }


            // Perform forward propagation
            MLP_MSELIN_forward();

            // Prepare y_index (true labels for the current mini-batch)
            array<float, elem> y_index;
            for(int k=0; k<elem; ++k) {
                int data_idx_in_shuffled_set = current_minibatch_indices[k];
                 if (data_idx_in_shuffled_set < num_train) {
                    y_index[k] = y[data_idx_in_shuffled_set];
                } else {
                    y_index[k] = 0.0f; // Pad if index was invalid
                }
            }

            // Calculate cost for the current mini-batch
            // Note: If last batch was padded, MLP_MSE_cost needs to be aware or use actual_batch_size.
            // Current MLP_MSE_cost uses y_true.size() which is 'elem'. So it's okay.
            float step_cost = MLP_MSE_cost(y_index);
            cost.push_back(step_cost);

            // Print progress
            // Optional: reduce frequency if num_training_runs > 1 or e is large
            if ( (e % 10 == 0 && (m_idx == 0 || m_idx == minibatches -1)) || e == epochs) {
                 printf("Epoch %d/%d, minibatch %04d/%04d, Loss (MSE) %g\n", e, epochs, m_idx + 1, minibatches, step_cost);
            }


            // Perform backpropagation to compute gradients
            MLP_MSELIN_backprop(y_index);

            // Calculate scaled weight updates (delta_W1 = eta * dL_dW1)
            array<array<float, n_features+1>, n_hidden> delta_W1;
            for (int i = 0; i < n_hidden; ++i) {
                for (int j = 0; j < n_features+1; ++j) {
                    delta_W1[i][j] = eta * delta_W1_unscaled[i][j];
                }
            }

            // Calculate scaled weight updates (delta_W2 = eta * dL_dW2)
            array<array<float, n_hidden+1>, n_output> delta_W2;
            for (int i = 0; i < n_output; ++i) {
                for (int j = 0; j < n_hidden+1; ++j) {
                    delta_W2[i][j] = eta * delta_W2_unscaled[i][j];
                }
            }

            // Update weights W1: W1 = W1 - delta_W1
            for (int i = 0; i < n_hidden; ++i) {
                for (int j = 0; j < n_features+1; ++j) {
                    w1[i][j] -= delta_W1[i][j];
                }
            }

            // Update weights W2: W2 = W2 - delta_W2
            for (int i = 0; i < n_output; ++i) {
                for (int j = 0; j < n_hidden+1; ++j) {
                    w2[i][j] -= delta_W2[i][j];
                }
            }
        } // End of mini-batch loop
    } // End of epoch loop
    // Removed: printf("Training completed.\n");
    // Removed: save_weights_w1();
    // Removed: save_weights_w2();
}


/**
 * @brief Predicts outputs for a given set of input samples.
 *        Processes data in chunks of size `elem` (mini-batch size used for global arrays).
 * @param x_ptr Pointer to the input data array (flattened, tot_elem * n_features).
 * @param y_pred_ptr Pointer to the output array where predictions will be stored (tot_elem).
 * @param tot_elem Total number of samples to predict.
 */
void MLP_MSELIN_predict(float* x_ptr, float* y_pred_ptr, int tot_elem) {
    int num_chunks = (tot_elem + elem - 1) / elem; // Ceiling division
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int current_batch_start_idx = chunk_idx * elem;
        int current_batch_size = std::min(elem, tot_elem - current_batch_start_idx);

        if (current_batch_size <= 0) break; 

        // Load data for the current chunk into x_input
        for (int k = 0; k < current_batch_size; ++k) {
            for (int f = 0; f < n_features; ++f) {
                x_input[k][f] = x_ptr[(current_batch_start_idx + k) * n_features + f];
            }
        }
        // Pad the rest of x_input for this batch if current_batch_size < elem
        for (int k = current_batch_size; k < elem; ++k) {
            for (int f = 0; f < n_features; ++f) {
                x_input[k][f] = 0.0f; // Pad with zeros
            }
        }
        
        MLP_MSELIN_forward(); // Processes the full 'elem' sized x_input

        // Copy results for the current_batch_size
        for (int k = 0; k < current_batch_size; ++k) {
             if (current_batch_start_idx + k < tot_elem) { 
                y_pred_ptr[current_batch_start_idx + k] = rA2[0][k];
            }
        }
    }
}

// Function to predict a single sample using the trained MLP
float MLP_predict_single_sample(const array<float, n_features>& single_x_val) {
    array<float, n_features + 1> local_a0;
    array<float, n_hidden>       local_rZ1;
    array<float, n_hidden>       local_rA1;
    array<float, n_hidden + 1>   local_a1;
    array<float, n_output>       local_rZ2; // Should be local_rZ2 not local_a2 for pre-activation
    
    local_rZ1.fill(0.0f); 
    local_rA1.fill(0.0f); 
    local_rZ2.fill(0.0f); 
    local_a1.fill(0.0f); 
    local_a0.fill(0.0f); 

    local_a0[0] = 1.0f; // Bias for input layer
    for (int j = 0; j < n_features; ++j) local_a0[j + 1] = single_x_val[j];

    // Hidden layer pre-activation: rZ1 = w1 * a0
    for (int h = 0; h < n_hidden; ++h) {
        for (int f = 0; f < n_features + 1; ++f) {
            local_rZ1[h] += w1[h][f] * local_a0[f];         
        }
    }

    // Hidden layer activation: rA1 = ReLU(rZ1)
    for (int h = 0; h < n_hidden; ++h) local_rA1[h] = max(0.0f, local_rZ1[h]);

    // Prepare activation for output layer: a1 (add bias to rA1)
    local_a1[0] = 1.0f; // Bias for hidden layer
    for (int h = 0; h < n_hidden; ++h) local_a1[h + 1] = local_rA1[h];
    
    // Output layer pre-activation: rZ2 = w2 * a1
    // Assuming n_output = 1
    for (int h = 0; h < n_hidden + 1; ++h) { 
        local_rZ2[0] += w2[0][h] * local_a1[h];
    }

    // Output layer activation (linear): rA2 = rZ2
    return local_rZ2[0]; // Network output
}


const unsigned int RNG_SEED_BASE = 1234;

int main() {
    const int num_training_runs = 10; // Define N, the number of training runs
    std::vector<float> all_mse_train;
    std::vector<float> all_mse_test;

    // --- Load/Generate Dataset (once) ---
    array<array<float, n_features>, num_train> x_train_data_orig;
    array<float, num_train> y_train_data_orig;
    array<array<float, n_features>, num_test> x_test_data; // Test data doesn't change
    array<float, num_test> y_test_data;

    if (load_dataset){
        printf("Loading training and test data from files...\n");
        read_dataset_train(x_train_data_orig, y_train_data_orig);
        read_dataset_test(x_test_data, y_test_data);
        cout << "Dataset loaded from files." << endl;
    }
    else {
        printf("Generating training and test data...\n");
        sinc2D_gen(x_train_data_orig[0].data(), y_train_data_orig.data(), num_train);
        sinc2D_gen(x_test_data[0].data(), y_test_data.data(), num_test);
        cout << "Dataset generated." << endl;
    }

    for (int run_idx = 0; run_idx < num_training_runs; ++run_idx) {
        std::cout << "\n--- Starting Training Run " << run_idx + 1 << "/" << num_training_runs << " ---" << std::endl;
        
        unsigned int current_run_seed = RNG_SEED_BASE + run_idx;
        srand(current_run_seed); // Seed for rand() used in MLP_initialize_weights

        // --- Shuffle Training Data for current run ---
        array<int, num_train> shuffled_ind;
        iota(shuffled_ind.begin(), shuffled_ind.end(), 0); 
        default_random_engine generator(current_run_seed); // Seed for shuffle
        shuffle(shuffled_ind.begin(), shuffled_ind.end(), generator);

        array<array<float, n_features>, num_train> x_train_shuffled;
        array<float, num_train> y_train_shuffled;
        for (int i = 0; i < num_train; ++i) {
            x_train_shuffled[i] = x_train_data_orig[shuffled_ind[i]];
            y_train_shuffled[i] = y_train_data_orig[shuffled_ind[i]];
        }

        // --- Train MLP ---
        // MLP_MSELIN_train will call MLP_initialize_weights() internally
        MLP_MSELIN_train(x_train_shuffled, y_train_shuffled);
        std::cout << "Training for run " << run_idx + 1 << " completed." << std::endl;

        // --- Evaluate on Training Data for this run ---
        array<float, num_train> ytrain_pred; 
        MLP_MSELIN_predict(x_train_shuffled[0].data(), ytrain_pred.data(), num_train);

        float acc_train_sum_sq_diff = 0.0f;
        for (int i = 0; i < num_train; ++i) { 
            float error = y_train_shuffled[i] - ytrain_pred[i];
            acc_train_sum_sq_diff += error * error;
        }
        float mse_train_current_run = acc_train_sum_sq_diff / (2.0f * num_train);
        all_mse_train.push_back(mse_train_current_run);
        printf("Run %d: Training accuracy (MSE): %g\n", run_idx + 1, mse_train_current_run);

        // --- Evaluate on Test Data for this run ---
        array<float, num_test> ytest_pred; 
        MLP_MSELIN_predict(x_test_data[0].data(), ytest_pred.data(), num_test);

        float acc_test_sum_sq_diff = 0.0f;
        for (int i = 0; i < num_test; ++i) { 
            float error = y_test_data[i] - ytest_pred[i];
            acc_test_sum_sq_diff += error * error;
        }
        float mse_test_current_run = acc_test_sum_sq_diff / (2.0f * num_test);
        all_mse_test.push_back(mse_test_current_run);
        printf("Run %d: Test accuracy (MSE): %g\n", run_idx + 1, mse_test_current_run);
    }

    // --- Calculate and Print Statistics ---
    std::cout << "\n--- Statistics over " << num_training_runs << " runs ---" << std::endl;
    std::cout << std::fixed << std::setprecision(8); 

    auto calculate_and_print_stats_float = [](const std::vector<float>& data, const std::string& name) {
        if (data.empty()) {
            std::cout << name << " MSE: No data." << std::endl;
            return;
        }
        double sum = std::accumulate(data.begin(), data.end(), 0.0); // Use double for sum for precision
        double mean = sum / data.size();
        
        double sq_sum_diff = 0.0;
        for(float val_float : data) {
            double val_double = static_cast<double>(val_float);
            sq_sum_diff += (val_double - mean) * (val_double - mean);
        }
        double std_dev = (data.size() <= 1) ? 0.0 : std::sqrt(sq_sum_diff / (data.size() -1)); // Sample std dev
        if (data.size() == 1) std_dev = 0.0;

        float min_val_float = *std::min_element(data.begin(), data.end());
        float max_val_float = *std::max_element(data.begin(), data.end());

        std::cout << name << " MSE:" << std::endl;
        std::cout << "  Mean:    " << mean << std::endl;
        std::cout << "  Std Dev: " << std_dev << std::endl;
        std::cout << "  Min:     " << static_cast<double>(min_val_float) << std::endl;
        std::cout << "  Max:     " << static_cast<double>(max_val_float) << std::endl;
        std::cout << "  Values:  [";
        for(size_t i=0; i<data.size(); ++i) {
            std::cout << static_cast<double>(data[i]) << (i == data.size()-1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    };

    calculate_and_print_stats_float(all_mse_train, "Training");
    calculate_and_print_stats_float(all_mse_test, "Test");

    // --- Save weights from the LAST run ---
    std::cout << "\nSaving weights from the final training run..." << std::endl;
    save_weights_w1();
    save_weights_w2();

    // --- Inference on Example Inputs (using weights from the last run) ---
    std::cout << "\n--- Inference on Example Inputs (using weights from last run) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(5); 

    std::vector<std::pair<std::array<double, n_features>, std::string>> example_raw_inputs = {
        {{0.0, 0.0}, "(0,0)"},
        {{1.0, 0.0}, "(1,0)"},
        {{acos(-1.0)/2.0, acos(-1.0)/2.0}, "(pi/2, pi/2)"}, 
        {{-1.0, 2.0}, "(-1,2)"},
        {{3.14159, 0.0}, "(pi,0)"} 
    };

    for (const auto& raw_example : example_raw_inputs) {
        const auto& raw_input_vals = raw_example.first;
        const auto& input_label = raw_example.second;

        array<float, n_features> current_input_float; // Changed from _fp to _float
        current_input_float[0] = static_cast<float>(raw_input_vals[0]);
        current_input_float[1] = static_cast<float>(raw_input_vals[1]);

        double x1_double = raw_input_vals[0];
        double x2_double = raw_input_vals[1];
        double sinc_x1_true = (x1_double == 0.0) ? 1.0 : std::sin(x1_double) / x1_double;
        double sinc_x2_true = (x2_double == 0.0) ? 1.0 : std::sin(x2_double) / x2_double;
        double true_output_double = 10.0 * sinc_x1_true * sinc_x2_true;

        float prediction_float = MLP_predict_single_sample(current_input_float);
        
        double prediction_double = static_cast<double>(prediction_float);

        std::cout << "Input: " << input_label << " [" << x1_double << ", " << x2_double << "]"
                  << " -> True Output: " << true_output_double
                  << ", MLP Prediction: " << prediction_double
                  << ", Error: " << (true_output_double - prediction_double)
                  << std::endl;
    }

    return 0;
}