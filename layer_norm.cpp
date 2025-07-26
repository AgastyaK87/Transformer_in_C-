#include "layer_norm.h"
#include <cmath>
#include "utils.h"

LayerNorm::LayerNorm(int d_model, std::ifstream& weight_file)
    : d_model_(d_model), epsilon_(1e-5) {
    // Define the shapes
    gamma_ = Eigen::RowVectorXf(d_model);
    beta_ = Eigen::RowVectorXf(d_model);

    // Load from file
    load_matrix(gamma_, weight_file);
    load_matrix(beta_, weight_file);
}

Eigen::MatrixXf LayerNorm::forward(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf output(x.rows(), x.cols());

    // Process each row (each token's vector) independently
    for (int i = 0; i < x.rows(); ++i) {
        // 1. Calculate mean and variance of the current row
        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().mean();

        // 2. Normalize the row
        Eigen::RowVectorXf normalized_row = (x.row(i).array() - mean) / std::sqrt(variance + epsilon_);
           //EPSILON IS A SMALL VALUE FOR NONZERO DIVISON
        // 3. Apply scale (gamma) and shift (beta)
        output.row(i) = normalized_row.array() * gamma_.array() + beta_.array();
    }

    return output;
}