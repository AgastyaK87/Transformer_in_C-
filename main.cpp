#include <iostream>
#include <vector>
#include "input_layer.h"

int main() {
    // --- Hyperparameters ---
    const int VOCAB_SIZE = 1000;
    const int D_MODEL = 12; // Use a small dimension for easy printing
    const int MAX_SEQ_LEN = 50;

    // --- Create the layer ---
    InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN);

    // --- Create a sample input sentence (as token IDs) ---
    std::vector<int> sample_tokens = {15, 234, 512, 9, 87}; // e.g., "hello world from C++"

    // --- Perform the forward pass ---
    Eigen::MatrixXf final_output = input_layer.forward(sample_tokens);

    // --- Print the result ---
    std::cout << "\nInput Token IDs: ";
    for(int id : sample_tokens) {
        std::cout << id << " ";
    }
    std::cout << "\n\nOutput Matrix (Shape: " << final_output.rows() << "x" << final_output.cols() << "):\n";
    std::cout << final_output << std::endl;

    return 0;
}