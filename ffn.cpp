#include "ffn.h"
#include "utils.h"

FFN::FFN(int d_model, int d_ff, std::ifstream& weight_file){
    //Init weights and bias with random values
    W1_ = Eigen::MatrixXf(d_model, d_ff);
    b1_ = Eigen::RowVectorXf(d_ff);
    W2_ = Eigen::MatrixXf(d_ff, d_model);
    b2_ = Eigen::RowVectorXf(d_model);

    load_matrix(W1_, weight_file);
    load_matrix(b1_, weight_file); // Assuming load_matrix works for RowVector too
    load_matrix(W2_, weight_file);
    load_matrix(b2_, weight_file);
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