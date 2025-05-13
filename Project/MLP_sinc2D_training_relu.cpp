
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <limits>
#include <fstream>
#include <iomanip>
#include <numeric>
using namespace std;


const int num_train = 150*150;              // number of training pattern (put a square number here)
const int num_test = 2500;


// //////////////////////////////////////////// //
//                 MLP parameters               //
// //////////////////////////////////////////// //
const int n_output = 1;                     // Number of outputs
const int n_features = 2;                   // Number of input features
const int n_hidden = 300;                   // Number of neurons in the hidden layer
const int epochs = 500;                     // Number of epochs
float eta = 1e-6;                           // Learning rate
const int minibatches = 30;                 // Number of mini-batches

vector<float> cost;
array<array<float, n_features+1>, n_hidden> w1 = {};
array<array<float, n_hidden+1>, n_output> w2 = {};




// Global declaration of variable used in the train step
const int elem = (num_train + minibatches -1 )/minibatches;     // inputs used in each minibatch

// forward
array<array<float, n_features>, elem> x_input;
array<array<float, elem>, n_features> rA0;
array<array<float, elem>, n_features+1> a0;
array<array<float, elem>, n_hidden> rZ1;
array<array<float, elem>, n_hidden> rA1;
array<array<float, elem>, n_hidden+1> a1;
array<array<float, elem>, n_output> rZ2;
array<array<float, elem>, n_output> rA2;


// backpropagation
array<array<float, elem>, n_output> dL_dZ2;
array<array<float, n_hidden+1>, n_output> dL_dW2;
array<array<float, elem>, n_hidden+1> dL_dA1;
array<array<float, elem>, n_hidden> activation_prime_of_rZ1;
array<array<float, elem>, n_hidden> dL_drZ1;
array<array<float, n_features+1>, n_hidden> dL_dW1;
array<array<float, n_features+1>, n_hidden> delta_W1_unscaled;
array<array<float, n_hidden+1>, n_output> delta_W2_unscaled;






void A_mult_B(const float* A, const float* B, float* C,
              int rigA, int colA, int colB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < colB; j++) {
            C[i * colB + j] = 0.0;
            for (int k = 0; k < colA; k++) {
                C[i*colB+j] += A[i*colA+k] * B[k*colB+j];
            }
        }
    }
}


void A_mult_B_T(const float* A, const float* B, float* C,
                   int rigA, int colA, int rigB) {
    for (int i = 0; i < rigA; i++) {
        for (int j = 0; j < rigB; j++) {
            C[i * rigB + j] = 0.0;
            for (int k = 0; k < colA; k++) {
                C[i*rigB+j] += A[i*colA+k] * B[j*colA+k];
            }
        }
    }
}


void A_T_mult_B(const float* A, const float* B, float* C,
                int rigA, int colA, int colB) {

    for (int i = 0; i < colA; i++) {
        for (int j = 0; j < colB; j++) {
            C[i * colB + j] = 0.0;
            for (int k = 0; k < rigA; k++) {
                C[i * colB + j] += A[k * colA + i] * B[k * colB + j];
            }
        }
    }
}


void elem_mult_elem(const float* A, const float* B, float* C, int rig, int col) {
    for (int i = 0; i < rig; ++i) {
        for (int j = 0; j < col; ++j) {
            C[i*col+j] = A[i*col+j] * B[i*col+j];
        }
    }
}





//sinc2D function generation
void sinc2D_gen(float* x, float* y, int num_patterns){
    int num_points = sqrt(num_patterns);

    // linspace x1
    vector<float> x1(num_points);
    float start_x1 = -5.0;
    float end_x1 = 5.0;
    float step_x1 = (end_x1 - start_x1) / (num_points - 1);
    for (int i = 0; i < num_points; ++i){
        x1[i] = start_x1 + i * step_x1;
    }


    // linspace x2
    vector<float> x2(num_points);
    float start_x2 = -5.0;
    float end_x2 = 5.0;
    float step_x2 = (end_x2 - start_x2) / (num_points - 1);
    for (int i = 0; i < num_points; ++i){
        x2[i] = start_x2 + i * step_x2;
    }


    // meshgrid
    vector<vector<float>> XX1(num_points, vector<float>(num_points));
    vector<vector<float>> XX2(num_points, vector<float>(num_points));
    for (int i = 0; i < num_points; ++i){
        for (int j = 0; j < num_points; ++j){
            XX1[i][j] = x1[j];
            XX2[i][j] = x2[i];
        }
    }


    // sinc2D
    vector<vector<float>> YY(num_points, vector<float>(num_points));
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            float sinc_x1 = (XX1[i][j] == 0) ? 1.0 : sin(XX1[i][j]) / XX1[i][j];
            float sinc_x2 = (XX2[i][j] == 0) ? 1.0 : sin(XX2[i][j]) / XX2[i][j];
            YY[i][j] = 10.0 * sinc_x1 * sinc_x2;
        }
    }


    // initialization x e y
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            x[(i*num_points+j)*n_features] = XX1[j][i];
            x[(i*num_points+j)*n_features + 1] = XX2[j][i];
            y[i * num_points + j] = YY[j][i];
        }
    }
}




// ////////////////////////////////////////////////////////////////////////// //
// MODIFICATO <<< INIZIO SEZIONE FUNZIONI DI ATTIVAZIONE E LORO GRADIENTI >>> //
// ////////////////////////////////////////////////////////////////////////// //

// // This function computes the sigmoid function for a scalar, a vector or a matrix
// void MLP_sigmoid_inplace(const array<array<float, elem>, n_hidden> &z, array<array<float, elem>, n_hidden> &sig_out){
//     for (int i = 0; i < n_hidden; ++i) {
//         for (int j = 0; j < elem; ++j) {
//             sig_out[i][j] = 1.0 / (1.0 + exp(-z[i][j]));
//         }
//     }
// }

// // Compute sigmoid gradient
// void MLP_sigmoid_gradient_inplace(const array<array<float, elem>, n_hidden> &Z, array<array<float, elem>, n_hidden> &sigGrad_out) {
//     for (int i = 0; i < n_hidden; ++i) {
//         for (int j = 0; j < elem; ++j) {
//             float sigmoid_val = 1.0 / (1.0 + exp(-Z[i][j])); // Calcola il valore sigmoid
//             sigGrad_out[i][j] = sigmoid_val * (1.0 - sigmoid_val); // Calcola il gradiente
//         }
//     }
// }


// This function computes the ReLU function for a scalar, a vector or a matrix
void MLP_relu_inplace(const array<array<float, elem>, n_hidden> &z, array<array<float, elem>, n_hidden> &relu_out){
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            relu_out[i][j] = max(0.0f, z[i][j]); // ReLU(x) = max(0, x)
        }
    }
}

// Compute ReLU gradient
void MLP_relu_gradient_inplace(const array<array<float, elem>, n_hidden> &Z, array<array<float, elem>, n_hidden> &reluGrad_out) {
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < elem; ++j) {
            reluGrad_out[i][j] = (Z[i][j] > 0.0f) ? 1.0f : 0.0f; // Derivata: 1 se Z > 0, altrimenti 0
        }
    }
}
// //////////////////////////////////////////////////////////////////////// //
// MODIFICATO <<< FINE SEZIONE FUNZIONI DI ATTIVAZIONE E LORO GRADIENTI >>> //
// //////////////////////////////////////////////////////////////////////// //


// Compute the forward step
void MLP_MSELIN_forward(){
    // rA0: is the "reduced A0" and it coincides with the transpose of the tall input matrix (nObs x nInput)
    for (int i = 0; i < elem; ++i) {
        for (int j = 0; j < n_features; ++j) {
            rA0[j][i] = x_input[i][j];
        }
    }


    // MLP_extend
    // A0  = E(rA0) It is the "extended" version of rA0, obtained by it by adding a row of ones as its new first row
    // Extend matrix X by adding the bias
    for (int i = 0; i < n_features+1; ++i) {
        for (int j = 0; j < elem; ++j) {
            a0[i][j] = (i == 0) ? 1 : rA0[i-1][j];
        }
    }


    // rZ1 = \sum(W1,A0).  It is the pre-activation at layer 1 (the hidden one)
    A_mult_B(w1[0].data(), a0[0].data(), rZ1[0].data(), n_hidden, n_features+1, elem);


    // rA1 = \sigma(rZ1).  It is the output of the first layer (the hidden one)
    MLP_relu_inplace(rZ1, rA1);


    // MLP_extend
    // A1  = E(rA1).       It is the extended version of rA1
    // Extend matrix X by adding the bias
    for (int i = 0; i < n_hidden+1; ++i) {
        for (int j = 0; j < elem; ++j) {
            a1[i][j] = (i == 0) ? 1 : rA1[i-1][j];
        }
    }


    // rZ2 = \sum(W2,A1).  It is the pre-activation at layer 2 (the output one)
    A_mult_B(w2[0].data(), a1[0].data(), rZ2[0].data(), n_output, n_hidden+1, elem);


    rA2 = rZ2;
}



// initialize weights to Unif[-1,1]
void MLP_initialize_weights(){
    array<float, n_hidden*(n_features+1)> w1_temp;
    for(int i=0; i<n_hidden*(n_features+1); ++i){
        w1_temp[i] = 2*(static_cast<float>(rand())/RAND_MAX)-1;
    }

    // reshape
    int index = 0;
    for (int j = 0; j < (n_features+1); ++j) {              //col
        for (int i = 0; i < n_hidden; ++i) {                //row
            w1[i][j] = w1_temp[index++];
        }
    }


    array<float, n_output*(n_hidden+1)> w2_temp;
    for(int i=0; i<n_output*(n_hidden+1); ++i){
        w2_temp[i] = 2*(static_cast<float>(rand())/RAND_MAX)-1;
    }

    // reshape
    index = 0;
    for (int j = 0; j < (n_hidden+1); ++j) {
        for (int i = 0; i < n_output; ++i) {
            w2[i][j] = w2_temp[index++];
        }
    }
}




float MLP_MSE_cost(const array<float, elem> &y) {
    vector<float> diff(y.size());
    for (int i = 0; i < y.size(); ++i) {
        diff[i] = (y[i] - rA2[0][i]);
        diff[i] *= diff[i];
    }


    float cost = (accumulate(diff.begin(), diff.end(), 0.0))/ (2.0 * y.size());
    return cost;
}



// Compute the partial derivative of the loss with respect to the two weighting matrices W2 and W1, using the backpropagation algorithm.
void MLP_MSELIN_backprop(const array<float, elem> &y){ /*float l1, float l2)*/
    // rA2 is 1xB
    // A1  is (H+1)xB
    // A0  is (D+1)xB
    // rZ1 is HxB
    // y   is 1xB
    // W1  is Hx(D+1)
    // W2  is 1x(H+1)

    // Step 1: compute dL_dZ2 of size 1xB
    // NB: rA2 coincides with y_pred
    // NB: dL_dZe could be called "grad2", the gradient on the output layer with respect the pre-activation Z2
    for(int i = 0; i<n_output; i++){
        for(int j = 0; j < elem; ++j) {
            dL_dZ2[i][j] = rA2[i][j] - y[j];
        }
    }


    // Step 2: compute dL_dW2 % size 1x(H+1)
    // NB: dL_dW2 could be called "delta_W2_unscaled", because it is of the same size of W2 and stores the unscaled variation
    A_mult_B_T(dL_dZ2[0].data(), a1[0].data(), dL_dW2[0].data(), n_output, elem, n_hidden+1);


    // Step 3: compute dL_dA1 of size (H+1)xB
    A_T_mult_B(w2[0].data(), dL_dZ2[0].data(), dL_dA1[0].data(), n_output, n_hidden+1, elem);



    // Step 4: compute dL_drZ1 of size HxB (also sigma_prime_of_rZ1 has size HxB)
    // NB: dL_drZ1 could have been called "grad1", since it is the gradient at the first layer (the hidden one), with respect to Z1
    MLP_relu_gradient_inplace(rZ1, activation_prime_of_rZ1);
    elem_mult_elem(dL_dA1[1].data(), activation_prime_of_rZ1[0].data(),dL_drZ1[0].data(),  n_hidden, elem);



    // Step 5: compute dL_dW1 of size Hx(D+1)
    // NB: dL_dW1 could be called "delta_W1_unscaled", because it is of the same size of W2 and stores the unscaled variation of W1
    A_mult_B_T(dL_drZ1[0].data(), a0[0].data(), dL_dW1[0].data(), n_hidden, elem, n_features+1);



    // Step 6: regularise or not
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

// learn weights from training data
void MLP_MSELIN_train(const array<array<float, n_features>, num_train> &x, const array<float, num_train> &y){
    // initialize weights w1 and w2
    MLP_initialize_weights();


    cost.push_back(numeric_limits<float>::infinity());

    // loop: epochs
    for(int e=1; e<=epochs; e++) {


        //reshape
        array<array<int, elem>, minibatches> I;
        for (int i = 0; i < num_train; ++i) {
            int row = i % minibatches;
            int col = i / minibatches;
            I[row][col] = i;
        }



        // loop: minibatches
        for(int m=1; m<=minibatches; ++m){
            array<int, elem> idx = I[m-1];

            // Compute the forward step
            for(int i=0; i<elem; i++) {
                copy(x[idx[i]].begin(), x[idx[i]].end(), x_input[i].begin());
            }
            // Feedforward
            MLP_MSELIN_forward();


            // Compute cost function
            array<float, elem> y_index;
            for(int i=0; i<elem; i++) {
                y_index[i] = y[idx[i]];
            }
            float step_cost = MLP_MSE_cost(y_index);
            cost.push_back(step_cost);


            printf("Epoch %d/%d, minibatch %04d, Loss (MSE) %g\n", e, epochs, m, step_cost);


            // Compute gradient via backpropagation
            MLP_MSELIN_backprop(y_index);


            array<array<float, n_features+1>, n_hidden> delta_W1;
            for (int i = 0; i < n_hidden; ++i) {
                for (int j = 0; j < n_features+1; ++j) {
                    delta_W1[i][j] = eta * delta_W1_unscaled[i][j];
                }
            }

            array<array<float, n_hidden+1>, n_output> delta_W2;
            for (int i = 0; i < n_output; ++i) {
                for (int j = 0; j < n_hidden+1; ++j) {
                    delta_W2[i][j] = eta * delta_W2_unscaled[i][j];
                }
            }



            for (int i = 0; i < n_hidden; ++i) {
                for (int j = 0; j < n_features+1; ++j) {
                    w1[i][j] -= delta_W1[i][j];
                }
            }

            for (int i = 0; i < n_output; ++i) {
                for (int j = 0; j < n_hidden+1; ++j) {
                    w2[i][j] -= delta_W2[i][j];
                }
            }
        }

    }
    printf("Training completed.\n");
    // --- Save the trained weights to files ---
    ofstream w1_file("weights_w1.txt");
    if (w1_file.is_open()) {
        for (int i = 0; i < n_hidden; ++i) {
            for (int j = 0; j < n_features + 1; ++j) {
                w1_file << fixed << setprecision(8) << w1[i][j] << (j == n_features ? "" : " ");
            } // We use fixed and setprecision(8) to ensure a consistent number of decimal places are saved, which should be sufficient for our fixed-point conversion later.
            w1_file << endl;
        }
        w1_file.close();
        cout << "Weights w1 saved to weights_w1.txt" << endl;
    } else {
        cerr << "Unable to open file weights_w1.txt for writing." << endl;
    }

    ofstream w2_file("weights_w2.txt");
    if (w2_file.is_open()) {
        for (int i = 0; i < n_output; ++i) {
            for (int j = 0; j < n_hidden + 1; ++j) {
                w2_file << fixed << setprecision(8) << w2[i][j] << (j == n_hidden ? "" : " ");
            }
            w2_file << endl;
        }
        w2_file.close();
        cout << "Weights w2 saved to weights_w2.txt" << endl;
    } else {
        cerr << "Unable to open file weights_w2.txt for writing." << endl;
    }

}




/* Predict the outputs for all the observations in X, where each row of X is a distinct observation.*/
void MLP_MSELIN_predict(float* x, float* y, int tot_elem) {
    for (int i = 0; i < tot_elem; i += elem) {

        for (int k = 0; k < elem * n_features; ++k) {
            int row = k / n_features;
            int col = k % n_features;
            x_input[row][col] = x[i * n_features + k];
        }

        // Feedforward
        MLP_MSELIN_forward();

        // Copia dei risultati nel vettore y
        copy(rA2[0].begin(), rA2[0].begin() + elem, y + i);
    }
}




int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    array<array<float, n_features>, num_train> x_train;
    array<float, num_train> y_train;
    sinc2D_gen(x_train[0].data(), y_train.data(), num_train);

    array<array<float, n_features>, num_test> x_test;
    array<float, num_test> y_test;
    sinc2D_gen(x_test[0].data(), y_test.data(), num_test);



    // Shuffling training data
    array<int, num_train> shuffled_ind;
    for (int i = 0; i < num_train; ++i) {
        shuffled_ind[i] = i;
    }

    default_random_engine generator(std::time(nullptr));
    shuffle(shuffled_ind.begin(), shuffled_ind.end(), generator);

    array<array<float, n_features>, num_train> x_train_temp;
    array<float, num_train> y_train_temp;

    for (int i = 0; i < num_train; ++i) {
        x_train_temp[i] = x_train[shuffled_ind[i]];
        y_train_temp[i] = y_train[shuffled_ind[i]];
    }

    x_train = x_train_temp;
    y_train = y_train_temp;

    // Learn weights from training data
    MLP_MSELIN_train(x_train, y_train);



   /* Predict the outputs for all the observations in X */
    array<float, num_train> ytrain_pred;
    MLP_MSELIN_predict(x_train[0].data(), ytrain_pred.data(), num_train);

    array<float, num_test> ytest_pred;
    MLP_MSELIN_predict(x_test[0].data(), ytest_pred.data(), num_test);



    // Compute accuracy (MSE)
    float acc_train = 0.0;
    for (int i = 0; i < y_train.size(); ++i) {
        acc_train += (y_train[i] - ytrain_pred[i])*(y_train[i] - ytrain_pred[i]);
    }
    acc_train /= (2.0 * y_train.size());
    printf("Training accuracy (MSE): %g\n", acc_train);

    float acc_test = 0.0;
    for (int i = 0; i < y_test.size(); ++i) {
        acc_test += (y_test[i] - ytest_pred[i])*(y_test[i] - ytest_pred[i]);
    }
    acc_test /= (2.0 * y_test.size());
    printf("Test accuracy: (MSE): %g\n", acc_test);


    return 0;
}