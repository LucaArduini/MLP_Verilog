// The main purpose of this script is to train the neural network using fixed-point numbers 
// and generate a grid of prediction data, which is then saved to a text file.
// A separate Python script will use this file to visually plot the function learned by the network.

#include <iostream>                 // For std::cout, std::cerr, std::endl (console output, file error handling)
#include <vector>                   // For std::vector (used in sinc2D_gen, MLP_MSE_cost)
#include <array>                    // For std::array (used for weight matrices, activations, data)
#include <cmath>                    // For std::pow, std::sqrt, std::exp (commented out), std::sin, acos
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

vector<double> cost;                 // Store cost as double for accurate reporting

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
            temp_accumulator_fp tmp_sum = temp_accumulator_fp{0.0};
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
            temp_accumulator_fp tmp_sum = temp_accumulator_fp{0.0};
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
            temp_accumulator_fp tmp_sum = temp_accumulator_fp{0.0};
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
            temp_accumulator_fp tmp_sum = temp_accumulator_fp{0.0};
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
    fixed_point_16 start_x1 = fixed_point_16{-5.0}; fixed_point_16 end_x1 = fixed_point_16{5.0};
    auto step_x1 = (num_points == 1) ? fixed_point_16{0.0f} : static_cast<fixed_point_16>((end_x1 - start_x1) / (num_points - 1));
    for (int i = 0; i < num_points; ++i) x1_coords[i] = start_x1 + i * step_x1;

    vector<fixed_point_16> x2_coords(num_points);
    fixed_point_16 start_x2 = fixed_point_16{-5.0}; fixed_point_16 end_x2 = fixed_point_16{5.0};
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
            reluGrad_out[i][j] = (Z[i][j] > fixed_point_16{0.0f}) ? fixed_point_16{1.0f} : fixed_point_16{0.0f};
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

    // Step 2: Compute dL_dW2_wide (Q(2*I+F).(2*F)) -> Q15.16 if inputs are Q7.8. Or more general with template.
    // For Q7.8 * Q7.8 = Q14.16. Sums can increase integer part.
    // temp_accumulator_fp is Q(63-16).16 = Q47.16. Sufficient.
    A_mult_B_T_wide_dest(dL_dZ2[0].data(), a1[0].data(), dL_dW2_wide[0].data(), n_output, elem, n_hidden+1);
    
    // Step 3: Compute dL_dA1_wide (temp_accumulator_fp)
    array<array<temp_accumulator_fp, elem>, n_hidden + 1> dL_dA1_temp_wide; 
    A_T_mult_B_wide_dest(w2[0].data(), dL_dZ2[0].data(), dL_dA1_temp_wide[0].data(), n_output, n_hidden + 1, elem);

    // Step 4: Compute dL_drZ1 (Q7.8)
    MLP_relu_gradient_inplace(rZ1, activation_prime_of_rZ1); 
    
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            // dL_dA1_temp_wide[i+1][j] is temp_accumulator_fp
            // activation_prime_of_rZ1[i][j] is fixed_point_16
            // Product is temp_accumulator_fp. Then cast to fixed_point_16.
            temp_accumulator_fp temp_prod = dL_dA1_temp_wide[i+1][j] * static_cast<temp_accumulator_fp>(activation_prime_of_rZ1[i][j]);
            dL_drZ1[i][j] = static_cast<fixed_point_16>(temp_prod);
        }
    }
    
    // Step 5: Compute dL_dW1_wide (temp_accumulator_fp)
    A_mult_B_T_wide_dest(dL_drZ1[0].data(), a0[0].data(), dL_dW1_wide[0].data(), n_hidden, elem, n_features+1);

    // Step 6: Convert wide gradients to Q7.8 with clipping
    const temp_accumulator_fp grad_clip_abs_val_wide = temp_accumulator_fp{100.0}; 

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
    MLP_initialize_weights();

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
            if (e % 10 == 0 && (m == 1 || m == minibatches) ) { 
                 printf("Epoch %d/%d, minibatch %04d, Loss (MSE) %g\n", e, epochs, m, step_cost_double);
            }

            MLP_MSELIN_backprop(y_index);

            const fixed_point_16 max_abs_final_update = fixed_point_16{2.0/256.0}; 

            array<array<fixed_point_16, n_features+1>, n_hidden> delta_W1;
            for (int i = 0; i < n_hidden; ++i) {
                for (int j = 0; j < n_features+1; ++j) {
                    // Product eta * delta_W1_unscaled[i][j] is fixed_point_16 (Q7.8 * Q7.8 = Q14.16, then cast to Q7.8)
                    // The result of multiplication is wider, then cast to fixed_point_16
                    temp_accumulator_fp wide_update = static_cast<temp_accumulator_fp>(eta) * delta_W1_unscaled[i][j];
                    fixed_point_16 update_val = static_cast<fixed_point_16>(wide_update);
                    
                    if (update_val > max_abs_final_update) update_val = max_abs_final_update;
                    else if (update_val < -max_abs_final_update) update_val = -max_abs_final_update;
                    delta_W1[i][j] = update_val;
                }
            }
            array<array<fixed_point_16, n_hidden+1>, n_output> delta_W2;
            for (int i = 0; i < n_output; ++i) {
                for (int j = 0; j < n_hidden+1; ++j) {
                    temp_accumulator_fp wide_update = static_cast<temp_accumulator_fp>(eta) * delta_W2_unscaled[i][j];
                    fixed_point_16 update_val = static_cast<fixed_point_16>(wide_update);

                    if (update_val > max_abs_final_update) update_val = max_abs_final_update;
                    else if (update_val < -max_abs_final_update) update_val = -max_abs_final_update;
                    delta_W2[i][j] = update_val;
                }
            }

            for (int i = 0; i < n_hidden; ++i) for (int j = 0; j < n_features+1; ++j) w1[i][j] -= delta_W1[i][j];
            for (int i = 0; i < n_output; ++i) for (int j = 0; j < n_hidden+1; ++j) w2[i][j] -= delta_W2[i][j];
        }
    }
    printf("Training completed.\n");
    // In this version of the file, there's no point in saving the weight matrices. Here, we train
    // the network, run our inference tests, and that's it—we're not interested in storing it.
    // save_weights_w1();
    // save_weights_w2();
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
        // Zero-pad if current_batch_size < elem
        for (int i = current_batch_size; i < elem; ++i) {
            for (int f = 0; f < n_features; ++f) {
                x_input[i][f] = fixed_point_16{0.0}; 
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

// Accumulator for single sample prediction, can be same as training if precision is sufficient
// Or can be different if specific inference precision is desired. Here, using the same.
using temp_accumulator_test = cnl::scaled_integer<int64_t, cnl::power<-fractional_bits*2>>; 

// Function to predict a single sample using the trained MLP (VERBOSE version)
fixed_point_16 MLP_predict_single_sample(const array<fixed_point_16, n_features>& single_x_val) {
    array<fixed_point_16, n_features + 1> local_a0;
    array<fixed_point_16, n_hidden>       local_rZ1;
    array<fixed_point_16, n_hidden>       local_rA1;
    array<fixed_point_16, n_hidden + 1>   local_a1;

    local_a0[0] = fixed_point_16{1.0f};
    for (int j = 0; j < n_features; ++j) local_a0[j + 1] = single_x_val[j];

    for (int h = 0; h < n_hidden; ++h) {
        temp_accumulator_test sum = temp_accumulator_test{0.0};
        for (int f = 0; f < n_features + 1; ++f) {
            sum += static_cast<temp_accumulator_test>(w1[h][f]) * local_a0[f];
        }
        local_rZ1[h] = static_cast<fixed_point_16>(sum);
    }

    for (int h = 0; h < n_hidden; ++h) local_rA1[h] = max(fixed_point_16{0.0f}, local_rZ1[h]);

    for (int h = 0; h < n_hidden; ++h) {
        std::cout << "a1[" << h << "] = " << cnl::unwrap(local_rA1[h]) << " (double: " << static_cast<double>(local_rA1[h]) << ")" << std::endl;
    }

    local_a1[0] = fixed_point_16{1.0f};
    for (int h = 0; h < n_hidden; ++h) local_a1[h + 1] = local_rA1[h];
    
    temp_accumulator_test sum_out = temp_accumulator_test{0.0};
    for (int h = 0; h < n_hidden + 1; ++h) { 
        sum_out += static_cast<temp_accumulator_test>(w2[0][h]) * local_a1[h];
        std::cout << "sum_out after adding w2[0][" << h << "] (" << static_cast<double>(w2[0][h]) 
                  << ") * local_a1[" << h << "] (" << static_cast<double>(local_a1[h]) << ") = " 
                  << static_cast<double>(sum_out) << " (raw: " << cnl::unwrap(sum_out) << ")" << std::endl;
    }
    return static_cast<fixed_point_16>(sum_out);
}

// QUIET version for grid generation
fixed_point_16 MLP_predict_single_sample_quiet(const array<fixed_point_16, n_features>& single_x_val) {
    array<fixed_point_16, n_features + 1> local_a0;
    array<fixed_point_16, n_hidden>       local_rZ1;
    array<fixed_point_16, n_hidden>       local_rA1;
    array<fixed_point_16, n_hidden + 1>   local_a1;

    local_a0[0] = fixed_point_16{1.0f}; 
    for (int j = 0; j < n_features; ++j) local_a0[j + 1] = single_x_val[j];

    for (int h = 0; h < n_hidden; ++h) {
        temp_accumulator_test sum = temp_accumulator_test{0.0}; 
        for (int f = 0; f < n_features + 1; ++f) {
            sum += static_cast<temp_accumulator_test>(w1[h][f]) * local_a0[f];
        }
        local_rZ1[h] = static_cast<fixed_point_16>(sum);
    }

    for (int h = 0; h < n_hidden; ++h) local_rA1[h] = max(fixed_point_16{0.0f}, local_rZ1[h]);

    local_a1[0] = fixed_point_16{1.0f}; 
    for (int h = 0; h < n_hidden; ++h) local_a1[h + 1] = local_rA1[h];
    
    temp_accumulator_test sum_out = temp_accumulator_test{0.0}; 
    for (int h = 0; h < n_hidden + 1; ++h) { 
        sum_out += static_cast<temp_accumulator_test>(w2[0][h]) * local_a1[h];
    }
    return static_cast<fixed_point_16>(sum_out);
}


// Define a global constant for the seed for clarity
const unsigned int RNG_SEED = 5134;

int main() {
    srand(RNG_SEED);
    array<array<fixed_point_16, n_features>, num_train> x_train_data;
    array<fixed_point_16, num_train> y_train_data;
    array<array<fixed_point_16, n_features>, num_test> x_test_data;
    array<fixed_point_16, num_test> y_test_data;
    if (load_dataset) {
        read_dataset_train(x_train_data, y_train_data);
        read_dataset_test(x_test_data, y_test_data);

        cout << "Dataset loaded from files." << endl;
    } else {
        cout << "Generating dataset..." << endl;
        sinc2D_gen(x_train_data[0].data(), y_train_data.data(), num_train, "train");
        sinc2D_gen(x_test_data[0].data(), y_test_data.data(), num_test, "test");
    }

    array<int, num_train> shuffled_ind;
    iota(shuffled_ind.begin(), shuffled_ind.end(), 0); 
    default_random_engine generator(RNG_SEED);
    shuffle(shuffled_ind.begin(), shuffled_ind.end(), generator);

    array<array<fixed_point_16, n_features>, num_train> x_train_shuffled;
    array<fixed_point_16, num_train> y_train_shuffled;
    for (int i = 0; i < num_train; ++i) {
        x_train_shuffled[i] = x_train_data[shuffled_ind[i]];
        y_train_shuffled[i] = y_train_data[shuffled_ind[i]];
    }

    MLP_MSELIN_train(x_train_shuffled, y_train_shuffled);

    array<fixed_point_16, num_train> ytrain_pred;
    MLP_MSELIN_predict(x_train_shuffled[0].data(), ytrain_pred.data(), num_train);

    array<fixed_point_16, num_test> ytest_pred;
    MLP_MSELIN_predict(x_test_data[0].data(), ytest_pred.data(), num_test);

    double acc_train_sum_sq_diff = 0.0;
    for (int i = 0; i < num_train; ++i) { 
        fixed_point_16 error = y_train_shuffled[i] - ytrain_pred[i];
        acc_train_sum_sq_diff += static_cast<double>(error) * static_cast<double>(error);
    }
    double mse_train = acc_train_sum_sq_diff / (2.0 * num_train);
    printf("Training accuracy (MSE): %g\n", mse_train);

    double acc_test_sum_sq_diff = 0.0;
    for (int i = 0; i < num_test; ++i) { 
        fixed_point_16 error = y_test_data[i] - ytest_pred[i];
        acc_test_sum_sq_diff += static_cast<double>(error) * static_cast<double>(error);
    }
    double mse_test = acc_test_sum_sq_diff / (2.0 * num_test);
    printf("Test accuracy (MSE): %g\n", mse_test);

    std::cout << "\n--- Inference on Example Inputs ---" << std::endl;
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

        std::cout << "\nRunning VERBOSE prediction for " << input_label << std::endl;
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

    // --- Generate Grid Data for Plotting ---
    std::cout << "\n--- Generating Grid for Plotting ---" << std::endl;
    const int grid_points_per_dim = 100; 
    const double grid_min = -5.0;
    const double grid_max = 5.0;
    std::ofstream plot_data_file("inference_grid_results.txt");

    if (!plot_data_file.is_open()) {
        std::cerr << "Error: Could not open inference_grid_results.txt for writing." << std::endl;
    } else {
        plot_data_file << std::fixed << std::setprecision(8); 

        std::vector<double> x_coords(grid_points_per_dim);
        double step = (grid_points_per_dim == 1) ? 0.0 : (grid_max - grid_min) / (grid_points_per_dim - 1);

        for (int i = 0; i < grid_points_per_dim; ++i) {
            x_coords[i] = grid_min + i * step;
        }

        for (int i = 0; i < grid_points_per_dim; ++i) { // Iterates over x2 coordinates
            for (int j = 0; j < grid_points_per_dim; ++j) { // Iterates over x1 coordinates
                double current_x1_double = x_coords[j]; 
                double current_x2_double = x_coords[i]; 

                array<fixed_point_16, n_features> current_input_fp;
                current_input_fp[0] = static_cast<fixed_point_16>(current_x1_double);
                current_input_fp[1] = static_cast<fixed_point_16>(current_x2_double);
                
                fixed_point_16 prediction_fp = MLP_predict_single_sample_quiet(current_input_fp);
                double prediction_double = static_cast<double>(prediction_fp);

                plot_data_file << current_x1_double << " "
                               << current_x2_double << " "
                               << prediction_double << std::endl;
            }
        }
        plot_data_file.close();
        if (plot_data_file.fail()) {
            std::cerr << "Error occurred during writing to inference_grid_results.txt." << std::endl;
        } else {
            std::cout << "Inference grid data (" << grid_points_per_dim << "x" << grid_points_per_dim 
                      << " points) saved to inference_grid_results.txt" << std::endl;
        }
    }
    
    return 0;
}