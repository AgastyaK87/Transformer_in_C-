#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <Eigen/Dense>

class LayerNorm {
public:
    // Constructor
    LayerNorm(int d_model);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    int d_model_;
    float epsilon_;

    // Learnable parameters for scaling and shifting
    Eigen::RowVectorXf gamma_;
    Eigen::RowVectorXf beta_;
};

#endif //LAYER_NORM_H