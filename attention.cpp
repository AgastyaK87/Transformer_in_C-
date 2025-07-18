#include "attention.h"
#include <cmath>
#include <iostream>

SingleHeadAttention::SingleHeadAttention(int d_model, int d_k) : d_k_(d_k) {
    // Initialize weight matrices with random values.
    // Dimensions are (d_model, d_k) to project input vectors
    // from d_model down to the head dimension d_k.
    W_q_ = Eigen::MatrixXf::Random(d_model, d_k);
    W_k_ = Eigen::MatrixXf::Random(d_model, d_k);
    W_v_ = Eigen::MatrixXf::Random(d_model, d_k);
}

// Applies the softmax function row-wise to a matrix.
void SingleHeadAttention::softmax(Eigen::MatrixXf& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        // Find the max value in the row for numerical stability
        float max_val = matrix.row(i).maxCoeff();
        // Subtract max, take exponent, and calculate sum
        matrix.row(i) = (matrix.row(i).array() - max_val).exp();
        float row_sum = matrix.row(i).sum();
        // Divide by the sum to get probabilities
        matrix.row(i) /= row_sum;
    }
}

Eigen::MatrixXf SingleHeadAttention::forward(const Eigen::MatrixXf& x) {
    // 1. Calculate Q, K, V matrices by projecting the input.
    // x has shape (seq_len, d_model)
    // W matrices have shape (d_model, d_k)
    // Resulting Q, K, V have shape (seq_len, d_k)
    Eigen::MatrixXf Q = x * W_q_;
    Eigen::MatrixXf K = x * W_k_;
    Eigen::MatrixXf V = x * W_v_;

    // 2. Calculate attention scores (unnormalized).
    // Q has shape (seq_len, d_k)
    // K.transpose() has shape (d_k, seq_len)
    // Resulting scores have shape (seq_len, seq_len)
    Eigen::MatrixXf scores = (Q * K.transpose()) / std::sqrt(static_cast<float>(d_k_));

    // 3. Apply softmax to scores to get attention weights.
    // This converts scores into a probability distribution.
    softmax(scores);

    // 4. Calculate the final output.
    // scores has shape (seq_len, seq_len)
    // V has shape (seq_len, d_k)
    // Resulting output has shape (seq_len, d_k)
    Eigen::MatrixXf output = scores * V;

    return output;
}

//Multihead Attention Implementation

MultiHeadAttention::MultiHeadAttention(int n_heads, int d_model, int d_k)
    : n_heads_(n_heads), d_k_(d_k) {

    // 1. Create the committee of attention heads
    for (int i = 0; i < n_heads_; ++i) {
        heads_.emplace_back(d_model, d_k_);
    }

    // 2. Initialize the final output weight matrix.
    // This matrix projects the concatenated outputs of all heads
    // back down to the model's original dimension (d_model).
    // The input dimension to this layer is (n_heads * d_k).
    W_o_ = Eigen::MatrixXf::Random(n_heads_ * d_k_, d_model);
}

Eigen::MatrixXf MultiHeadAttention::forward(const Eigen::MatrixXf& x) {
    //1. Run all attention heads in parallel and collect outputs
    std::vector<Eigen::MatrixXf> head_outputs;
    for (auto& head : heads_) {
        head_outputs.push_back(head.forward(x));
    }

    //2. Concat the outputs of all the heads
    // Create matrix to hold head outputs side by side
    int seq_len = x.rows();
    Eigen::MatrixXf concatenated_output(seq_len, n_heads_ * d_k_);

    for (int i = 0; i<n_heads_; ++i) {
        //Eigen's block method lets us copy small matrix (head output)
        //THis gets copied into bigger matrix (concat otutput)
        concatenated_output.block(0, i * d_k_, seq_len, d_k_) = head_outputs[i];
    }

    // 3. Apply the final linear layer (W_o).
    // This combines the insights from all heads into a single unified output.
    // concatenated_output has shape (seq_len, n_heads * d_k)
    // W_o_ has shape (n_heads * d_k, d_model)
    // Final output has shape (seq_len, d_model)
    Eigen::MatrixXf final_output = concatenated_output * W_o_;

    return final_output;
}
