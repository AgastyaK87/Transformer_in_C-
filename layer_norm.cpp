#include "layer_norm.h"
#include <cmath>

LayerNorm::LayerNorm(int d_model) : d_model_(d_model), epsilon_(1e-5) {
    // Initialize gamma to ones and beta to zeros.
    // This makes the initial normalization have no effect.
    gamma_ = Eigen::RowVectorXf::Ones(d_model);
    beta_ = Eigen::RowVectorXf::Zero(d_model);
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