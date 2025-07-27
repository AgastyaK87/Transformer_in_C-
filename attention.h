#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>
#include <fstream>

class SingleHeadAttention {
public:
    // Constructor

    SingleHeadAttention(int d_model, int d_k, std::ifstream& weight_file);

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

class MultiHeadAttention
{
public:
    MultiHeadAttention(int n_heads, int d_model, int d_k, std::ifstream& weight_file);

    //Forward Pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x, const Eigen::MatrixXf& encoder_k, const Eigen::MatrixXf& encoder_v);
private:
    int n_heads_;
    int d_k_;

    //Vector to hold all individual attention heads
    std::vector<SingleHeadAttention> heads_;

    //Linear Layer to combine head outputs
    Eigen::MatrixXf W_o_;

};

#endif //ATTENTION_H