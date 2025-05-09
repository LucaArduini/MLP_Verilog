// mlp_fixed.h
#ifndef MLP_FIXED_H
#define MLP_FIXED_H

#include <array>
#include <vector>
#include <cmath>
#include <string>
#include "mlp_config.h"

typedef int fixed_point_32;
const int FRACTIONAL_BITS = 16;

fixed_point_32 float_to_fixed(float val);
float fixed_to_float(fixed_point_32 val);

std::array<fixed_point_32, n_output> fixed_MLP_inference(const std::array<float, n_features>& x,
                                                      const std::array<std::array<fixed_point_32, n_features + 1>, n_hidden>& w1_fixed,
                                                      const std::array<std::array<fixed_point_32, n_hidden + 1>, n_output>& w2_fixed);

bool load_fixed_point_weights(const std::string& filename_w1, const std::string& filename_w2,
                                  std::array<std::array<fixed_point_32, n_features + 1>, n_hidden>& w1_fixed,
                                  std::array<std::array<fixed_point_32, n_hidden + 1>, n_output>& w2_fixed);

#endif // MLP_FIXED_H