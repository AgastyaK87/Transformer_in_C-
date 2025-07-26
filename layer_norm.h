#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <Eigen/Dense>
#include <fstream>

class LayerNorm {
public:
    // Constructor
    LayerNorm(int d_model, std::ifstream& weight_file);

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