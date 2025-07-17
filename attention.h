#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>

class SingleHeadAttention {
public:
    // Constructor
    SingleHeadAttention(int d_model, int d_k);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    // Helper function for softmax
    void softmax(Eigen::MatrixXf& matrix);

    int d_k_; // Dimension of key/query/value vectors
    Eigen::MatrixXf W_q_; // Query weight matrix
    Eigen::MatrixXf W_k_; // Key weight matrix
    Eigen::MatrixXf W_v_; // Value weight matrix
};

#endif //ATTENTION_H