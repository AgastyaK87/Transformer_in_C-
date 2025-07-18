#include "ffn.h"

FFN::FFN(int d_model, int d_ff){
    //Init weights and bias with random values
    W1_ = Eigen::MatrixXf::Random(d_model, d_ff);
    b1_ = Eigen::RowVectorXf::Random(d_ff);
    W2_ = Eigen::MatrixXf::Random(d_ff, d_model);
    b2_ = Eigen::RowVectorXf::Random(d_model);
}

// ReLU activation function: max(0, x)
Eigen::MatrixXf FFN::relu(const Eigen::MatrixXf& x) {
    return x.array().cwiseMax(0);
}

Eigen::MatrixXf FFN::forward(const Eigen::MatrixXf& x) {
    // 1. First linear transformation
    Eigen::MatrixXf output = x * W1_;
    output.rowwise() += b1_; // Add bias to each row

    // 2. Apply ReLU activation
    output = relu(output);

    // 3. Second linear transformation
    output = output * W2_;
    output.rowwise() += b2_; // Add bias to each row

    return output;

    //Do this for amt of layers
}