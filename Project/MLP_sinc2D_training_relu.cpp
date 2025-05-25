#include <iostream>                 // For std::cout, std::cerr, std::endl (console output, file error handling)
#include <vector>                   // For std::vector (used in sinc2D_gen, MLP_MSE_cost)
#include <array>                    // For std::array (used for weight matrices, activations, data)
#include <cmath>                    // For std::pow, std::sqrt, std::exp (commented out), std::sin
#include <algorithm>                // For std::shuffle, std::max, std::copy
#include <random>                   // For std::default_random_engine (shuffling)
#include <ctime>                    // For std::time (seeding for rand and random_engine)
#include <limits>                   // For std::numeric_limits (cost initialization)
#include <fstream>                  // For std::ofstream (writing weight files)
#include <iomanip>                  // For std::fixed, std::setprecision (decimal output formatting)
#include <numeric>                  // For std::accumulate (cost calculation)

#include "cnl/include/cnl/all.h"    // For cnl::scaled_integer and related functionalities (fixed_point_16)

using namespace std;
namespace impl = cnl::_impl;        // Namespace alias for CNL implementation details



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
float eta = 1e-6;                   // Learning rate
const int minibatches = 30;         // Number of mini-batches for training

vector<float> cost;                 // Vector to store the cost (MSE) at each training step (per mini-batch)

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
void sinc2D_gen(float* x, float* y, int num_patterns){
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
    for (int col = 0; col < num_points; ++col) {                // Outer loop iterates over 'j' in original
        for (int row = 0; row < num_points; ++row) {            // Inner loop iterates over 'i' in original
            int pattern_idx = col * num_points + row;
            x[pattern_idx * n_features + 0] = XX1[row][col];    // XX1[i][j] from original indexing
            x[pattern_idx * n_features + 1] = XX2[row][col];    // XX2[i][j] from original indexing
            y[pattern_idx]                  = YY[row][col];     // YY[i][j] from original indexing
        }
    }
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

    // Initialize cost history with a very high value
    cost.push_back(numeric_limits<float>::infinity());


    // Main training loop over epochs
    for(int e=1; e<=epochs; e++) {

        // Prepare mini-batch indices
        // I[m][k] will be the k-th global index for the m-th minibatch.
        // This creates `minibatches` number of index arrays, each of size `elem`.
        array<array<int, elem>, minibatches> I;
        for (int i = 0; i < num_train; ++i) {
            int row = i % minibatches;
            int col = i / minibatches;
            I[row][col] = i;
        }

        // Loop over mini-batches
        for(int m=1; m<=minibatches; ++m){
            array<int, elem> idx = I[m-1];      // Get indices for the current mini-batch

            // Prepare x_input for the current mini-batch
            // x_input is (elem x n_features)
            for(int i=0; i<elem; i++) {
                copy(x[idx[i]].begin(), x[idx[i]].end(), x_input[i].begin());
            }

            // Perform forward propagation
            MLP_MSELIN_forward();

            // Prepare y_index (true labels for the current mini-batch)
            array<float, elem> y_index;
            for(int i=0; i<elem; i++) {
                y_index[i] = y[idx[i]];         // Store cost
            }

            // Calculate cost for the current mini-batch
            float step_cost = MLP_MSE_cost(y_index);
            cost.push_back(step_cost);

            // Print progress
            printf("Epoch %d/%d, minibatch %04d, Loss (MSE) %g\n", e, epochs, m, step_cost);


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
    printf("Training completed.\n");

    // Save the trained weights
    save_weights_w1();
    save_weights_w2();
}


/**
 * @brief Predicts outputs for a given set of input samples.
 *        Processes data in chunks of size `elem` (mini-batch size used for global arrays).
 * @param x Pointer to the input data array (flattened, tot_elem * n_features).
 * @param y Pointer to the output array where predictions will be stored (tot_elem).
 * @param tot_elem Total number of samples to predict.
 */
void MLP_MSELIN_predict(float* x, float* y_pred, int tot_elem) {
    /* Predict the outputs for all the observations in X, where each row of X is a distinct observation.*/
    
    for (int i = 0; i < tot_elem; i += elem) {
        // Load current chunk of data into x_input
        // x_input is (elem x n_features)
        for (int k = 0; k < elem * n_features; ++k) {
            int row = k / n_features;
            int col = k % n_features;
            x_input[row][col] = x[i * n_features + k];
        }

        // Perform forward propagation
        MLP_MSELIN_forward();

        // Copia dei risultati nel vettore y_pred
        copy(rA2[0].begin(), rA2[0].begin() + elem, y_pred + i);
    }
}




int main() {
    // Seed the random number generator for weight initialization and shuffling
    srand(static_cast<unsigned int>(time(nullptr)));

    // --- Generate Training Data ---
    // Declare arrays to hold training data
    array<array<float, n_features>, num_train> x_train;
    array<float, num_train> y_train;
    sinc2D_gen(x_train[0].data(), y_train.data(), num_train);

    // --- Generate Test Data ---
    array<array<float, n_features>, num_test> x_test;
    array<float, num_test> y_test;
    sinc2D_gen(x_test[0].data(), y_test.data(), num_test);


    // --- Shuffle Training Data ---
    // Create an array of indices from 0 to num_train-1
    array<int, num_train> shuffled_ind;
    for (int i = 0; i < num_train; ++i) {
        shuffled_ind[i] = i;
    }

    // Shuffle the indices
    default_random_engine generator(std::time(nullptr));            // Seed random engine
    shuffle(shuffled_ind.begin(), shuffled_ind.end(), generator);

    // Create temporary arrays to store shuffled data
    array<array<float, n_features>, num_train> x_train_temp;
    array<float, num_train> y_train_temp;

    // Populate shuffled arrays using the shuffled indices
    for (int i = 0; i < num_train; ++i) {
        x_train_temp[i] = x_train[shuffled_ind[i]];
        y_train_temp[i] = y_train[shuffled_ind[i]];
    }

    x_train = x_train_temp;
    y_train = y_train_temp;


    // --- Train the MLP ---
    // Learn weights from training data
    MLP_MSELIN_train(x_train, y_train);


    // --- Make Predictions ---
    // Predict on training data
    array<float, num_train> ytrain_pred;
    MLP_MSELIN_predict(x_train[0].data(), ytrain_pred.data(), num_train);

    // Predict on test data
    array<float, num_test> ytest_pred;
    MLP_MSELIN_predict(x_test[0].data(), ytest_pred.data(), num_test);


    // --- Compute Accuracy (MSE) ---
    // Training accuracy
    float acc_train = 0.0;
    for (int i = 0; i < y_train.size(); ++i) {
        acc_train += (y_train[i] - ytrain_pred[i])*(y_train[i] - ytrain_pred[i]);
    }
    acc_train /= (2.0 * y_train.size());
    printf("Training accuracy (MSE): %g\n", acc_train);

    // Test accuracy
    float acc_test = 0.0;
    for (int i = 0; i < y_test.size(); ++i) {
        acc_test += (y_test[i] - ytest_pred[i])*(y_test[i] - ytest_pred[i]);
    }
    acc_test /= (2.0 * y_test.size());
    printf("Test accuracy: (MSE): %g\n", acc_test);

    return 0;
}