// This script is used for statistical analysis of the MLP's training performance.
// It trains the network multiple times to calculate the average, minimum, maximum,
// and standard deviation of the final training and test errors.
//
// These statistics will be compared against the results from a similar training
// script that uses floating-point instead of fixed-point arithmetic.
// This comparison helps identify the most reliable training approach to generate
// the final weights for hardware inference.

#include <iostream>                 // For std::cout, std::cerr, std::endl (console output, file error handling)
#include <vector>                   // For std::vector (used in sinc2D_gen, MLP_MSE_cost)
#include <array>                    // For std::array (used for weight matrices, activations, data)
#include <cmath>                    // For std::pow, std::sqrt, std::exp (commented out), std::sin, acos
#include <algorithm>                // For std::shuffle, std::max, std::copy, std::min_element, std::max_element
#include <random>                   // For std::default_random_engine (shuffling)
#include <ctime>                    // For std::time (seeding for rand and random_engine)
#include <limits>                   // For std::numeric_limits (cost initialization)
#include <fstream>                  // For std::ofstream (writing weight files)
#include <iomanip>                  // For std::fixed, std::setprecision (decimal output formatting)
#include <numeric>                  // For std::accumulate, std::iota, std::inner_product

#include "../cnl/include/cnl/all.h" // For cnl::scaled_integer and related functionalities (fixed_point_16)

using namespace std;
namespace impl = cnl::_impl;        // Namespace alias for CNL implementation details

bool load_dataset = true; // Set to true to load dataset from files

// //////////////////////////////////////////// //
//          Fixed-Point parameters            //
// //////////////////////////////////////////// //

// Use one consistent constant for fractional bits
const int fractional_bits = 8; 

// Define the fixed-point type using CNL library
// int16_t: underlying integer type (16 bits)
// cnl::power<-fractional_bits>: specifies the position of the binary point (8 fractional bits)
using fixed_point_16 = cnl::scaled_integer<int16_t, cnl::power<-fractional_bits>>;

// Define a wider accumulator type for intermediate sums in matrix multiplications
using temp_accumulator_fp = cnl::scaled_integer<int64_t, cnl::power<-fractional_bits*2>>; 

// If you want to use a wider type for training, say Q15.16, these are the critical values to update:

// const int fractional_bits = 16;
// using fixed_point_16 = cnl::scaled_integer<int32_t, cnl::power<-fractional_bits>>;
// using temp_accumulator_fp = cnl::scaled_integer<int64_t, cnl::power<-fractional_bits>>; 
// fixed_point_16 eta = fixed_point_16{0.001}; 
// const temp_accumulator_fp grad_clip_abs_val_wide = 1000.0;
// const fixed_point_16 max_abs_final_update = fixed_point_16{0.001};

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
fixed_point_16 eta = fixed_point_16{1.0/256.0}; // Learning rate (smallest positive step for Q7.8)
const int minibatches = 30;         // Number of mini-batches for training

vector<double> cost;                 // Store cost as double for accurate reporting (cleared per training run)

// Weight matrices for the MLP
array<array<fixed_point_16, n_features+1>, n_hidden> w1 = {};
array<array<fixed_point_16, n_hidden+1>, n_output> w2 = {};


// Global declaration of variables used in the train step
const int elem = (num_train + minibatches -1 )/minibatches;     // Inputs used in each minibatch

// //////////////////////////////////////////// //
//      Global arrays for Forward Propagation   //
// //////////////////////////////////////////// //
array<array<fixed_point_16, n_features>, elem> x_input;
array<array<fixed_point_16, elem>, n_features> rA0;
array<array<fixed_point_16, elem>, n_features+1> a0;
array<array<fixed_point_16, elem>, n_hidden> rZ1;
array<array<fixed_point_16, elem>, n_hidden> rA1;
array<array<fixed_point_16, elem>, n_hidden+1> a1;
array<array<fixed_point_16, elem>, n_output> rZ2;
array<array<fixed_point_16, elem>, n_output> rA2;

// //////////////////////////////////////////// //
//    Global arrays for Backpropagation         //
// //////////////////////////////////////////// //
array<array<fixed_point_16, elem>, n_output> dL_dZ2;
// Store unscaled gradients in wider type to prevent premature saturation
array<array<temp_accumulator_fp, n_hidden+1>, n_output> dL_dW2_wide;
array<array<temp_accumulator_fp, n_features+1>, n_hidden> dL_dW1_wide;
// These will hold the final Q7.8 unscaled gradients after potential clipping/scaling from wide type
array<array<fixed_point_16, n_features+1>, n_hidden> delta_W1_unscaled;
array<array<fixed_point_16, n_hidden+1>, n_output> delta_W2_unscaled;

// dL_dA1 is not stored globally in its wide form, only as fixed_point_16 if needed,
// but its wide temporary is used in backprop.
// array<array<fixed_point_16, elem>, n_hidden+1> dL_dA1; // This might not be needed if dL_dA1_temp_wide is sufficient
array<array<fixed_point_16, elem>, n_hidden> activation_prime_of_rZ1;
array<array<fixed_point_16, elem>, n_hidden> dL_drZ1;


// //////////////////////////////////////////// //
//                 Utilities functions          //
// //////////////////////////////////////////// //

void A_mult_B(const fixed_point_16* A_ptr, const fixed_point_16* B_ptr, fixed_point_16* C_ptr,
              int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            temp_accumulator_fp tmp_sum = 0.0;
            for (int k = 0; k < colA; k++) {
                tmp_sum += static_cast<temp_accumulator_fp>(A_ptr[i*colA+k]) * B_ptr[k*colB+j];
            }
            C_ptr[i * colB + j] = static_cast<fixed_point_16>(tmp_sum);
        }
    }
}

// Modified to write to a temp_accumulator_fp destination C_ptr
void A_mult_B_wide_dest(const fixed_point_16* A_ptr, const fixed_point_16* B_ptr, temp_accumulator_fp* C_ptr,
                        int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            temp_accumulator_fp tmp_sum = 0.0;
            for (int k = 0; k < colA; k++) {
                tmp_sum += static_cast<temp_accumulator_fp>(A_ptr[i*colA+k]) * B_ptr[k*colB+j];
            }
            C_ptr[i*colB+j] = tmp_sum; // Store full precision sum
        }
    }
}


void A_mult_B_T_wide_dest(const fixed_point_16* A_ptr, const fixed_point_16* B_ptr, temp_accumulator_fp* C_ptr,
                          int rigA, int colA, int rigB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < rigB; j++) {
            temp_accumulator_fp tmp_sum = 0.0;
            for (int k = 0; k < colA; k++) {
                tmp_sum += static_cast<temp_accumulator_fp>(A_ptr[i*colA+k]) * B_ptr[j*colA+k]; // Corrected B indexing
            }
            C_ptr[i*rigB+j] = tmp_sum; // Store full precision sum
        }
    }
}

void A_T_mult_B_wide_dest(const fixed_point_16* A_ptr, const fixed_point_16* B_ptr, temp_accumulator_fp* C_ptr,
                          int rigA, int colA, int colB) {
    for (int i = 0; i < colA; i++) {
        for (int j = 0; j < colB; j++) {
            temp_accumulator_fp tmp_sum = 0.0;
            for (int k = 0; k < rigA; k++) {
                tmp_sum += static_cast<temp_accumulator_fp>(A_ptr[k*colA+i]) * B_ptr[k*colB+j];
            }
            C_ptr[i*colB+j] = tmp_sum; // Store full precision sum
        }
    }
}


void elem_mult_elem(const fixed_point_16* A_ptr, const fixed_point_16* B_ptr, fixed_point_16* C_ptr, int rig, int col) {
    for (int i = 0; i < rig; ++i) {
        for (int j = 0; j < col; ++j) {
            C_ptr[i*col+j] = A_ptr[i*col+j] * B_ptr[i*col+j];
        }
    }
}

void save_weights_w1() {
    ofstream outFile_dec("weights_w1_decFix.txt");
    ofstream outFile_bin("weights_w1_binFix.txt");
    if (!outFile_bin || !outFile_dec) {
        cerr << "Error opening file for writing w1" << endl;
        return;
    }
    const unsigned char* byte_ptr;
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_features + 1; ++j) {
            outFile_dec << fixed << setprecision(8) << w1[i][j] << (j == n_features ? "" : " ");
            fixed_point_16 fixed_val(w1[i][j]);
            byte_ptr = static_cast<const unsigned char*>(static_cast<const void*>(&fixed_val));
            for (int byte_idx = sizeof(fixed_val) - 1; byte_idx >= 0; --byte_idx) {
                for (int bit_pos = CHAR_BIT - 1; bit_pos >= 0; --bit_pos) {
                    outFile_bin << ((byte_ptr[byte_idx] >> bit_pos) & 1);
                }
            }
            outFile_bin << (j == n_features ? "" : " ");
        }
        outFile_dec << endl;
        outFile_bin << endl;
    }
    outFile_dec.close(); outFile_bin.close();
    if (outFile_dec.fail() || outFile_bin.fail()) std::cerr << "Error during writing to w1 files." << std::endl;
    else std::cout << "Weights w1 saved." << endl;
}

void save_weights_w2() {
    ofstream outFile_dec("weights_w2_decFix.txt");
    ofstream outFile_bin("weights_w2_binFix.txt");
    if (!outFile_bin || !outFile_dec) {
        cerr << "Error opening file for writing w2" << endl;
        return;
    }
    const unsigned char* byte_ptr;
    for (int i = 0; i < n_output; ++i) {
        for (int j = 0; j < n_hidden + 1; ++j) {
            outFile_dec << fixed << setprecision(8) << w2[i][j] << (j == n_hidden ? "" : " ");
            fixed_point_16 fixed_val(w2[i][j]);
            byte_ptr = static_cast<const unsigned char*>(static_cast<const void*>(&fixed_val));
            for (int byte_idx = sizeof(fixed_val) - 1; byte_idx >= 0; --byte_idx) {
                for (int bit_pos = CHAR_BIT - 1; bit_pos >= 0; --bit_pos) {
                    outFile_bin << ((byte_ptr[byte_idx] >> bit_pos) & 1);
                }
            }
            outFile_bin << (j == n_hidden ? "" : " ");
        }
        outFile_dec << endl;
        outFile_bin << endl;
    }
    outFile_dec.close(); outFile_bin.close();
    if (outFile_dec.fail() || outFile_bin.fail()) std::cerr << "Error during writing to w2 files." << std::endl;
    else std::cout << "Weights w2 saved." << endl;
}



//Reminder: This function generates a dataset for f(x1, x2) = 10 * sinc(x1) * sinc(x2)
// where sinc(x) = sin(x)/x, and sinc(0) = 1.
void sinc2D_gen(fixed_point_16* x_ptr, fixed_point_16* y_ptr, int num_patterns, string type){
    int num_points = sqrt(num_patterns);
    vector<fixed_point_16> x1_coords(num_points);
    fixed_point_16 start_x1 = -5.0; fixed_point_16 end_x1 = 5.0;
    auto step_x1 = (num_points == 1) ? fixed_point_16{0.0f} : static_cast<fixed_point_16>((end_x1 - start_x1) / (num_points - 1));
    for (int i = 0; i < num_points; ++i) x1_coords[i] = start_x1 + i * step_x1;

    vector<fixed_point_16> x2_coords(num_points);
    fixed_point_16 start_x2 = -5.0; fixed_point_16 end_x2 = 5.0;
    auto step_x2 = (num_points == 1) ? fixed_point_16{0.0f} : static_cast<fixed_point_16>((end_x2 - start_x2) / (num_points - 1));
    for (int i = 0; i < num_points; ++i) x2_coords[i] = start_x2 + i * step_x2;

    vector<vector<fixed_point_16>> XX1(num_points, vector<fixed_point_16>(num_points));
    vector<vector<fixed_point_16>> XX2(num_points, vector<fixed_point_16>(num_points));
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            XX1[i][j] = x1_coords[j]; XX2[i][j] = x2_coords[i];
        }
    }
    vector<vector<fixed_point_16>> YY(num_points, vector<fixed_point_16>(num_points));
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            fixed_point_16 val_x1 = XX1[i][j]; fixed_point_16 val_x2 = XX2[i][j];
            double d_val_x1 = static_cast<double>(val_x1);
            double d_val_x2 = static_cast<double>(val_x2);
            double sinc_x1_d = (d_val_x1 == 0.0) ? 1.0 : std::sin(d_val_x1) / d_val_x1;
            double sinc_x2_d = (d_val_x2 == 0.0) ? 1.0 : std::sin(d_val_x2) / d_val_x2;
            YY[i][j] = static_cast<fixed_point_16>(10.0 * sinc_x1_d * sinc_x2_d);
        }
    }
    for (int col = 0; col < num_points; ++col) {
        for (int row = 0; row < num_points; ++row) {
            int pattern_idx = col * num_points + row;
            x_ptr[pattern_idx * n_features + 0] = XX1[row][col];
            x_ptr[pattern_idx * n_features + 1] = XX2[row][col];
            y_ptr[pattern_idx]                  = YY[row][col];
        }
    }
    //Save the dataset to files
    ofstream x_file("sinc2D_x_"+type+".txt");
    ofstream y_file("sinc2D_y_"+type+".txt");
    if (x_file.is_open() && y_file.is_open()) {
        for (int i = 0; i < num_patterns; ++i) {
            x_file << fixed << setprecision(8) << x_ptr[i * n_features] << " " << x_ptr[i * n_features + 1] << endl;
            y_file << fixed << setprecision(8) << y_ptr[i] << endl;
        }
        x_file.close();
        y_file.close();
        cout << "Dataset saved to sinc2D_x_" << type << ".txt and sinc2D_y_" << type << ".txt" << endl;
    } else {
        cerr << "Unable to open files for writing." << endl;
    }

}

// Function to read the dataset from files
void read_dataset_train(array<array<fixed_point_16, n_features>, num_train> &x, array<fixed_point_16, num_train> &y) {
    ifstream x_file("sinc2D_x_train.txt");
    ifstream y_file("sinc2D_y_train.txt");
    if (!x_file.is_open() || !y_file.is_open()) {
        cerr << "Error opening dataset files." << endl;
        return;
    }
    for (int i = 0; i < num_train; ++i) {
        x_file >> x[i][0] >> x[i][1];
        y_file >> y[i];
    }
    x_file.close(); y_file.close();
}

void read_dataset_test(array<array<fixed_point_16, n_features>, num_test> &x, array<fixed_point_16, num_test> &y) {
    ifstream x_file("sinc2D_x_test.txt");
    ifstream y_file("sinc2D_y_test.txt");
    if (!x_file.is_open() || !y_file.is_open()) {
        cerr << "Error opening dataset files." << endl;
        return;
    }
    for (int i = 0; i < num_test; ++i) {
        x_file >> x[i][0] >> x[i][1];
        y_file >> y[i];
    }
    x_file.close(); y_file.close();
}

void MLP_relu_inplace(const array<array<fixed_point_16, elem>, n_hidden> &z, array<array<fixed_point_16, elem>, n_hidden> &relu_out){
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            relu_out[i][j] = max(fixed_point_16{0.0f}, z[i][j]);
        }
    }
}

void MLP_relu_gradient_inplace(const array<array<fixed_point_16, elem>, n_hidden> &Z, array<array<fixed_point_16, elem>, n_hidden> &reluGrad_out) {
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            reluGrad_out[i][j] = (Z[i][j] > 0.0f) ? fixed_point_16{1.0f} : fixed_point_16{0.0f};
        }
    }
}

void MLP_MSELIN_forward(){
    for (int i = 0; i < elem; ++i) for (int j = 0; j < n_features; ++j) rA0[j][i] = x_input[i][j];
    for (int i = 0; i < n_features+1; ++i) for (int j = 0; j < elem; ++j) a0[i][j] = (i == 0) ? fixed_point_16{1.0f} : rA0[i-1][j];
    A_mult_B(w1[0].data(), a0[0].data(), rZ1[0].data(), n_hidden, n_features+1, elem);
    MLP_relu_inplace(rZ1, rA1);
    for (int i = 0; i < n_hidden+1; ++i) for (int j = 0; j < elem; ++j) a1[i][j] = (i == 0) ? fixed_point_16{1.0f} : rA1[i-1][j];
    A_mult_B(w2[0].data(), a1[0].data(), rZ2[0].data(), n_output, n_hidden+1, elem);
    rA2 = rZ2;
}

void MLP_initialize_weights(){
    array<fixed_point_16, n_hidden*(n_features+1)> w1_temp;
    for(int i=0; i<n_hidden*(n_features+1); ++i) w1_temp[i] = static_cast<fixed_point_16>(2.0 * (static_cast<double>(rand())/RAND_MAX) - 1.0);
    int index = 0; for (int j = 0; j < (n_features+1); ++j) for (int i = 0; i < n_hidden; ++i) w1[i][j] = w1_temp[index++];
    array<fixed_point_16, n_output*(n_hidden+1)> w2_temp;
    for(int i=0; i<n_output*(n_hidden+1); ++i) w2_temp[i] = static_cast<fixed_point_16>(2.0 * (static_cast<double>(rand())/RAND_MAX) - 1.0);
    index = 0; for (int j = 0; j < (n_hidden+1); ++j) for (int i = 0; i < n_output; ++i) w2[i][j] = w2_temp[index++];
}

double MLP_MSE_cost(const array<fixed_point_16, elem> &y_true) {
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        fixed_point_16 error_fixed = (y_true[i] - rA2[0][i]);
        double error_double = static_cast<double>(error_fixed);
        sum_sq_diff += error_double * error_double;
    }
    return sum_sq_diff / (2.0 * y_true.size());
}

void MLP_MSELIN_backprop(const array<fixed_point_16, elem> &y_true){
    // Step 1: Compute dL_dZ2 (Q7.8)
    for(int i = 0; i<n_output; i++){
        for(int j = 0; j < elem; ++j) {
            dL_dZ2[i][j] = rA2[i][j] - y_true[j];
        }
    }

    // Step 2: Compute dL_dW2_wide (Q23.8)
    A_mult_B_T_wide_dest(dL_dZ2[0].data(), a1[0].data(), dL_dW2_wide[0].data(), n_output, elem, n_hidden+1);
    
    // Step 3: Compute dL_dA1_wide (Q23.8)
    array<array<temp_accumulator_fp, elem>, n_hidden + 1> dL_dA1_temp_wide; 
    A_T_mult_B_wide_dest(w2[0].data(), dL_dZ2[0].data(), dL_dA1_temp_wide[0].data(), n_output, n_hidden + 1, elem);

    // Step 4: Compute dL_drZ1 (Q7.8)
    MLP_relu_gradient_inplace(rZ1, activation_prime_of_rZ1); 
    
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            dL_drZ1[i][j] = static_cast<fixed_point_16>(dL_dA1_temp_wide[i+1][j]) * activation_prime_of_rZ1[i][j];
        }
    }
    
    // Step 5: Compute dL_dW1_wide (Q23.8)
    A_mult_B_T_wide_dest(dL_drZ1[0].data(), a0[0].data(), dL_dW1_wide[0].data(), n_hidden, elem, n_features+1);

    // Step 6: Convert wide gradients to Q7.8 with clipping
    // This is crucial to prevent saturation in fixed-point representation
    const temp_accumulator_fp grad_clip_abs_val_wide = 100.0; //TO BE EXPERIMENTED WITH

    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_features+1; ++j) {
            temp_accumulator_fp val = dL_dW1_wide[i][j];
            if (val > grad_clip_abs_val_wide) val = grad_clip_abs_val_wide;
            else if (val < -grad_clip_abs_val_wide) val = -grad_clip_abs_val_wide;
            delta_W1_unscaled[i][j] = static_cast<fixed_point_16>(val);
        }
    }
    for (int i = 0; i < n_output; ++i) { 
        for (int j = 0; j < n_hidden+1; ++j) {
            temp_accumulator_fp val = dL_dW2_wide[i][j];
            if (val > grad_clip_abs_val_wide) val = grad_clip_abs_val_wide;
            else if (val < -grad_clip_abs_val_wide) val = -grad_clip_abs_val_wide;
            delta_W2_unscaled[i][j] = static_cast<fixed_point_16>(val);
        }
    }
}

void MLP_MSELIN_train(const array<array<fixed_point_16, n_features>, num_train> &x, const array<fixed_point_16, num_train> &y){
    cost.clear(); // Clear global cost vector for this new training run
    MLP_initialize_weights(); // Re-initialize weights for each training run

    for(int e=1; e<=epochs; e++) {
        array<array<int, elem>, minibatches> I;
        for (int i = 0; i < num_train; ++i) { int row = i % minibatches; int col = i / minibatches; I[row][col] = i; }

        for(int m=1; m<=minibatches; ++m){
            array<int, elem> idx = I[m-1];
            for(int i=0; i<elem; i++) copy(x[idx[i]].begin(), x[idx[i]].end(), x_input[i].begin());
            MLP_MSELIN_forward();
            array<fixed_point_16, elem> y_index;
            for(int i=0; i<elem; i++) y_index[i] = y[idx[i]];
            
            double step_cost_double = MLP_MSE_cost(y_index);
            cost.push_back(step_cost_double);
            if (e % 10 == 0 && (m == 1 || m == minibatches) ) { // Print less frequently
                 // Optional: Add run number if printing per-run training progress
                 // printf("Run %d, Epoch %d/%d, minibatch %04d, Loss (MSE) %g\n", current_run_number, e, epochs, m, step_cost_double);
                 // For now, keep it as is to reduce verbosity during multiple runs.
                 printf("Epoch %d/%d, minibatch %04d, Loss (MSE) %g\n", e, epochs, m, step_cost_double);
            }

            MLP_MSELIN_backprop(y_index);


            // Very small, controlled updates to prevent saturation
            // This is a crucial step to ensure the updates are within the range of fixed_point_16
            const fixed_point_16 max_abs_final_update = fixed_point_16{2.0/256.0}; // To be experimented with

            array<array<fixed_point_16, n_features+1>, n_hidden> delta_W1;
            for (int i = 0; i < n_hidden; ++i) {
                for (int j = 0; j < n_features+1; ++j) {
                    fixed_point_16 update_val = eta * delta_W1_unscaled[i][j];
                    if (update_val > max_abs_final_update) update_val = max_abs_final_update;
                    else if (update_val < -max_abs_final_update) update_val = -max_abs_final_update;
                    delta_W1[i][j] = update_val;
                }
            }
            array<array<fixed_point_16, n_hidden+1>, n_output> delta_W2;
            for (int i = 0; i < n_output; ++i) {
                for (int j = 0; j < n_hidden+1; ++j) {
                    fixed_point_16 update_val = eta * delta_W2_unscaled[i][j];
                    if (update_val > max_abs_final_update) update_val = max_abs_final_update;
                    else if (update_val < -max_abs_final_update) update_val = -max_abs_final_update;
                    delta_W2[i][j] = update_val;
                }
            }

            for (int i = 0; i < n_hidden; ++i) for (int j = 0; j < n_features+1; ++j) w1[i][j] -= delta_W1[i][j];
            for (int i = 0; i < n_output; ++i) for (int j = 0; j < n_hidden+1; ++j) w2[i][j] -= delta_W2[i][j];
        }
    }
    // Removed: printf("Training completed.\n");
    // Removed: save_weights_w1();
    // Removed: save_weights_w2();
}

void MLP_MSELIN_predict(fixed_point_16* x_ptr, fixed_point_16* y_pred_ptr, int tot_elem) {
    int num_chunks = (tot_elem + elem - 1) / elem;
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int current_batch_start_idx = chunk_idx * elem;
        int current_batch_size = std::min(elem, tot_elem - current_batch_start_idx);

        if (current_batch_size <= 0) break;

        for (int i = 0; i < current_batch_size; ++i) {
            for (int f = 0; f < n_features; ++f) {
                x_input[i][f] = x_ptr[(current_batch_start_idx + i) * n_features + f];
            }
        }
        for (int i = current_batch_size; i < elem; ++i) {
            for (int f = 0; f < n_features; ++f) {
                x_input[i][f] = 0.0; 
            }
        }
        
        MLP_MSELIN_forward();

        for (int i = 0; i < current_batch_size; ++i) {
             if (current_batch_start_idx + i < tot_elem) {
                y_pred_ptr[current_batch_start_idx + i] = rA2[0][i];
            }
        }
    }
}

using temp_accumulator_test = cnl::scaled_integer<int64_t, cnl::power<-fractional_bits*2>>; // 

// Function to predict a single sample using the trained MLP
fixed_point_16 MLP_predict_single_sample(const array<fixed_point_16, n_features>& single_x_val) {
    array<fixed_point_16, n_features + 1> local_a0;
    array<fixed_point_16, n_hidden>       local_rZ1;
    array<fixed_point_16, n_hidden>       local_rA1;
    array<fixed_point_16, n_hidden + 1>   local_a1;
    array<fixed_point_16, n_output>       local_rZ2;

    local_a0[0] = 1.0f;
    for (int j = 0; j < n_features; ++j) local_a0[j + 1] = single_x_val[j];

    for (int h = 0; h < n_hidden; ++h) {
        temp_accumulator_test sum = 0.0;
        for (int f = 0; f < n_features + 1; ++f) {
            sum += static_cast<temp_accumulator_test>(w1[h][f]) * local_a0[f];
        }
        local_rZ1[h] = static_cast<fixed_point_16>(sum);
    }

    for (int h = 0; h < n_hidden; ++h) local_rA1[h] = max(fixed_point_16{0.0f}, local_rZ1[h]);

    // Commented out verbose debug output for cleaner multi-run output
    // for (int h = 0; h < n_hidden; ++h) {
    //     std::cout << "a1[" << h << "] = " << cnl::unwrap(local_rA1[h]) << std::endl;
    // }

    local_a1[0] = 1.0f;
    for (int h = 0; h < n_hidden; ++h) local_a1[h + 1] = local_rA1[h];
    
    temp_accumulator_test sum_out = 0.0;
    for (int h = 0; h < n_hidden + 1; ++h) { 
        sum_out += static_cast<temp_accumulator_test>(w2[0][h]) * local_a1[h];
        // Commented out verbose debug output
        // std::cout << cnl::unwrap(sum_out) << " sum_out after adding w2[0][" << h << "] = " << cnl::unwrap(w2[0][h]) << " * local_a1[" << h << "] = " << cnl::unwrap(local_a1[h]) << std::endl;
    }
    local_rZ2[0] = static_cast<fixed_point_16>(sum_out);

    return local_rZ2[0];
}

// Define a global constant for the seed for clarity
const unsigned int RNG_SEED_BASE = 5134;

int main() {
    const int num_training_runs = 10; // Define N, the number of training runs
    std::vector<double> all_mse_train;
    std::vector<double> all_mse_test;

    // --- Load Dataset (once) ---
    array<array<fixed_point_16, n_features>, num_train> x_train_data_orig;
    array<fixed_point_16, num_train> y_train_data_orig;
    array<array<fixed_point_16, n_features>, num_test> x_test_data; // Test data doesn't need shuffling per run
    array<fixed_point_16, num_test> y_test_data;

    if (load_dataset) {
        read_dataset_train(x_train_data_orig, y_train_data_orig);
        read_dataset_test(x_test_data, y_test_data);
        cout << "Dataset loaded from files." << endl;
    } else {
        cout << "Generating dataset..." << endl;
        sinc2D_gen(x_train_data_orig[0].data(), y_train_data_orig.data(), num_train, "train");
        sinc2D_gen(x_test_data[0].data(), y_test_data.data(), num_test, "test");
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

        array<array<fixed_point_16, n_features>, num_train> x_train_shuffled;
        array<fixed_point_16, num_train> y_train_shuffled;
        for (int i = 0; i < num_train; ++i) {
            x_train_shuffled[i] = x_train_data_orig[shuffled_ind[i]];
            y_train_shuffled[i] = y_train_data_orig[shuffled_ind[i]];
        }

        // --- Train MLP ---
        // MLP_MSELIN_train will call MLP_initialize_weights() internally
        MLP_MSELIN_train(x_train_shuffled, y_train_shuffled);
        std::cout << "Training for run " << run_idx + 1 << " completed." << std::endl;


        // --- Evaluate on Training Data for this run ---
        array<fixed_point_16, num_train> ytrain_pred; // Declare locally for this run
        MLP_MSELIN_predict(x_train_shuffled[0].data(), ytrain_pred.data(), num_train);

        double acc_train_sum_sq_diff = 0.0;
        for (int i = 0; i < num_train; ++i) { 
            fixed_point_16 error = y_train_shuffled[i] - ytrain_pred[i];
            acc_train_sum_sq_diff += static_cast<double>(error) * static_cast<double>(error);
        }
        double mse_train_current_run = acc_train_sum_sq_diff / (2.0 * num_train);
        all_mse_train.push_back(mse_train_current_run);
        printf("Run %d: Training accuracy (MSE): %g\n", run_idx + 1, mse_train_current_run);

        // --- Evaluate on Test Data for this run ---
        array<fixed_point_16, num_test> ytest_pred; // Declare locally for this run
        MLP_MSELIN_predict(x_test_data[0].data(), ytest_pred.data(), num_test);

        double acc_test_sum_sq_diff = 0.0;
        for (int i = 0; i < num_test; ++i) { 
            fixed_point_16 error = y_test_data[i] - ytest_pred[i];
            acc_test_sum_sq_diff += static_cast<double>(error) * static_cast<double>(error);
        }
        double mse_test_current_run = acc_test_sum_sq_diff / (2.0 * num_test);
        all_mse_test.push_back(mse_test_current_run);
        printf("Run %d: Test accuracy (MSE): %g\n", run_idx + 1, mse_test_current_run);
    }

    // --- Calculate and Print Statistics ---
    std::cout << "\n--- Statistics over " << num_training_runs << " runs ---" << std::endl;
    std::cout << std::fixed << std::setprecision(8); // Set precision for stats output

    auto calculate_and_print_stats = [](const std::vector<double>& data, const std::string& name) {
        if (data.empty()) {
            std::cout << name << " MSE: No data." << std::endl;
            return;
        }
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        double mean = sum / data.size();
        
        double sq_sum_diff = 0.0;
        for(double val : data) {
            sq_sum_diff += (val - mean) * (val - mean);
        }
        double std_dev = (data.size() <= 1) ? 0.0 : std::sqrt(sq_sum_diff / (data.size() -1)); // Sample std dev
        if (data.size() == 1) std_dev = 0.0; // Avoid NaN for single run

        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());

        std::cout << name << " MSE:" << std::endl;
        std::cout << "  Mean:    " << mean << std::endl;
        std::cout << "  Std Dev: " << std_dev << std::endl;
        std::cout << "  Min:     " << min_val << std::endl;
        std::cout << "  Max:     " << max_val << std::endl;
        std::cout << "  Values:  [";
        for(size_t i=0; i<data.size(); ++i) {
            std::cout << data[i] << (i == data.size()-1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    };

    calculate_and_print_stats(all_mse_train, "Training");
    calculate_and_print_stats(all_mse_test, "Test");

    // --- Save weights from the LAST run (optional) ---
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

        array<fixed_point_16, n_features> current_input_fp;
        current_input_fp[0] = static_cast<fixed_point_16>(raw_input_vals[0]);
        current_input_fp[1] = static_cast<fixed_point_16>(raw_input_vals[1]);

        double x1_double = raw_input_vals[0];
        double x2_double = raw_input_vals[1];
        double sinc_x1_true = (x1_double == 0.0) ? 1.0 : std::sin(x1_double) / x1_double;
        double sinc_x2_true = (x2_double == 0.0) ? 1.0 : std::sin(x2_double) / x2_double;
        double true_output_double = 10.0 * sinc_x1_true * sinc_x2_true;

        fixed_point_16 prediction_fp = MLP_predict_single_sample(current_input_fp);
        double prediction_double = static_cast<double>(prediction_fp);

        std::cout << "Input: " << input_label << " [" << x1_double << ", " << x2_double << "]"
                  << " -> True Output: " << true_output_double
                  << ", MLP Prediction: " << prediction_double
                  << ", Error: " << (true_output_double - prediction_double)
                  << " (Q7.8: " << prediction_fp << ")"
                  << " Integer Representation: " << cnl::unwrap(prediction_fp)
                  << std::endl;
    }
    
    return 0;
}
