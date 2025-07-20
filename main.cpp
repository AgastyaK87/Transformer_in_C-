#include <iostream>
#include <vector>
#include "input_layer.h"
#include "encoder.h"

int main() {
    // --- Hyperparameters ---
    const int VOCAB_SIZE = 1000;
    const int D_MODEL = 512;
    const int MAX_SEQ_LEN = 50;
    const int N_HEADS = 8;
    const int D_FF = 2048;
    const int N_LAYERS = 6; // Standard number of encoder layers

    // --- Create the full model ---
    InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN);
    Encoder encoder(N_LAYERS, D_MODEL, N_HEADS, D_FF);

    // --- Create a sample input sentence ---
    std::vector<int> sample_tokens = {15, 234, 512, 9, 87, 34};

    // --- Perform the full forward pass ---
    // 1. Get input matrix from the input layer
    Eigen::MatrixXf x = input_layer.forward(sample_tokens);
    std::cout << "Input to Encoder shape: " << x.rows() << "x" << x.cols() << std::endl;

    // 2. Pass data through the complete N-layer encoder
    Eigen::MatrixXf final_output = encoder.forward(x);
    std::cout << "Output from Encoder shape: " << final_output.rows() << "x" << final_output.cols() << std::endl;

    std::cout << "\nCongratulations! You have successfully passed data through a complete Transformer Encoder." << std::endl;

    return 0;
}