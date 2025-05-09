// mlp_float.h
#ifndef MLP_FLOAT_H
#define MLP_FLOAT_H

#include <array>
#include <vector>
#include <cmath>
#include <string>
#include "mlp_config.h"

std::array<float, n_output> MLP_inference_float(const std::array<float, n_features>& x,
                                              const std::array<std::array<float, n_features + 1>, n_hidden>& w1_float,
                                              const std::array<std::array<float, n_hidden + 1>, n_output>& w2_float);

bool load_float_weights(const std::string& filename_w1, const std::string& filename_w2,
                        std::array<std::array<float, n_features + 1>, n_hidden>& w1_float,
                        std::array<std::array<float, n_hidden + 1>, n_output>& w2_float);

#endif // MLP_FLOAT_H