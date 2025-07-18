#include <iostream>
#include <vector>
#include "input_layer.h"
#include "attention.h"

int main() {
    // --- Hyperparameters ---
    const int VOCAB_SIZE = 1000;
    const int D_MODEL = 512;
    const int MAX_SEQ_LEN = 50;
    const int N_HEADS = 8; // Standard number of heads
    const int D_K = 64;   // Dimension per head (512 / 8 = 64)

    // --- Create the layers ---
    InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN);
    MultiHeadAttention multi_head_attention(N_HEADS, D_MODEL, D_K);

    // --- Create a sample input sentence ---
    std::vector<int> sample_tokens = {15, 234, 512, 9, 87};

    // --- Perform the full forward pass ---
    // 1. Get input matrix from the input layer
    Eigen::MatrixXf input_matrix = input_layer.forward(sample_tokens);
    std::cout << "\nInput Matrix Shape: " << input_matrix.rows() << "x" << input_matrix.cols() << std::endl;

    // 2. Feed the input matrix into the multi-head attention layer
    Eigen::MatrixXf attention_output = multi_head_attention.forward(input_matrix);
    std::cout << "Multi-Head Attention Output Shape: " << attention_output.rows() << "x" << attention_output.cols() << std::endl;

    std::cout << "\nSuccessfully passed data through the full Multi-Head Attention layer." << std::endl;

    return 0;
}