// verify_fixed_point.cpp
#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include "mlp_float.h"
#include "mlp_fixed.h"

using namespace std;

int main() {
    // Load floating-point weights
    array<array<float, n_features + 1>, n_hidden> w1_float;
    array<array<float, n_hidden + 1>, n_output> w2_float;
    if (!load_float_weights("weights_w1.txt", "weights_w2.txt", w1_float, w2_float)) {
        return 1;
    }

    // Load fixed-point weights
    array<array<fixed_point_32, n_features + 1>, n_hidden> w1_fixed;
    array<array<fixed_point_32, n_hidden + 1>, n_output> w2_fixed;
    if (!load_fixed_point_weights("weights_w1.txt", "weights_w2.txt", w1_fixed, w2_fixed)) {
        return 1;
    }

    // Prepare a dataset for comparison
    // Might also be loaded from a file or generated
    vector<array<float, n_features>> test_data = {
        {1.0, -2.0},
        {0.5, 1.5},
        {-1.0, -1.0},
        {2.0, 0.0},
        {0.0, 0.0},
        {-2.5, 3.1}
        // More test data to be added here
    };

    vector<float> float_predictions;
    vector<float> fixed_predictions_float;
    vector<float> absolute_errors;
    vector<float> relative_errors;

    cout << fixed << setprecision(8);

    for (const auto& input_sample : test_data) {
        // Run inference with floating-point model
        array<float, n_output> float_output = MLP_inference_float(input_sample, w1_float, w2_float);
        float_predictions.push_back(float_output[0]);

        // Run inference with fixed-point model
        array<fixed_point_32, n_output> fixed_output = fixed_MLP_inference(input_sample, w1_fixed, w2_fixed);
        fixed_predictions_float.push_back(fixed_to_float(fixed_output[0]));

        // Calculate errors
        float abs_err = abs(float_output[0] - fixed_predictions_float.back());
        absolute_errors.push_back(abs_err);

        float rel_err = 0.0;
        if (abs(float_output[0]) > 1e-8) { // Avoid division by very small numbers
            rel_err = abs_err / abs(float_output[0]);
        }
        relative_errors.push_back(rel_err);

        cout << "--- Input: [" << input_sample[0] << ", " << input_sample[1] << "] ---" << endl;
        cout << "  Float Prediction:        " << float_output[0] << endl;
        cout << "  Fixed Prediction (float): " << fixed_predictions_float.back() << endl;
        cout << "  Absolute Error:          " << abs_err << endl;
        cout << "  Relative Error:          " << rel_err << endl;
        cout << endl;
    }

    // Analyze the errors
    double mean_abs_error = accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / absolute_errors.size();
    double max_abs_error = *max_element(absolute_errors.begin(), absolute_errors.end());

    vector<float> finite_relative_errors;
    for (float err : relative_errors) {
        if (isfinite(err)) {
            finite_relative_errors.push_back(err);
        }
    }
    double mean_rel_error = finite_relative_errors.empty() ? 0.0 : accumulate(finite_relative_errors.begin(), finite_relative_errors.end(), 0.0) / finite_relative_errors.size();
    double max_rel_error = finite_relative_errors.empty() ? 0.0 : *max_element(finite_relative_errors.begin(), finite_relative_errors.end());

    sort(absolute_errors.begin(), absolute_errors.end());
    double percentile_90_abs_error = absolute_errors[static_cast<int>(0.9 * absolute_errors.size())];

    cout << "\n--- Error Analysis ---" << endl;
    cout << "Mean Absolute Error:          " << mean_abs_error << endl;
    cout << "Max Absolute Error:           " << max_abs_error << endl;
    cout << "Mean Relative Error (finite): " << mean_rel_error << endl;
    cout << "Max Relative Error (finite):  " << max_rel_error << endl;
    cout << "90th Percentile Absolute Error: " << percentile_90_abs_error << endl;

    return 0;
}