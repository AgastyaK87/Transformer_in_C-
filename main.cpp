#include <iostream>
#include <vector>
#include "input_layer.h"
#include "attention.h"

int main() {
    // --- Hyperparameters ---
    const int VOCAB_SIZE = 1000;
    const int D_MODEL = 512; // Standard dimension for a base model
    const int MAX_SEQ_LEN = 50;
    const int D_K = 64; // Standard dimension for a single attention head

    // --- Create the layers ---
    InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN);
    SingleHeadAttention attention_head(D_MODEL, D_K);

    // --- Create a sample input sentence ---
    std::vector<int> sample_tokens = {15, 234, 512, 9, 87};

    // --- Perform the full forward pass ---
    // 1. Get input matrix from the input layer
    Eigen::MatrixXf input_matrix = input_layer.forward(sample_tokens);
    std::cout << "\nInput Matrix Shape: " << input_matrix.rows() << "x" << input_matrix.cols() << std::endl;

    // 2. Feed the input matrix into the attention head
    Eigen::MatrixXf attention_output = attention_head.forward(input_matrix);
    std::cout << "Attention Output Matrix Shape: " << attention_output.rows() << "x" << attention_output.cols() << std::endl;

    std::cout << "\nAttention output:\n" << attention_output << std::endl;
    std::cout << "\nSuccessfully passed data through InputLayer and SingleHeadAttention." << std::endl;

    return 0;
}