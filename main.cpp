#include <iostream>
#include <vector>
#include <fstream>
#include "input_layer.h"
#include "encoder.h"
#include "decoder.h" // Include the new decoder header

int main() {
    // --- Hyperparameters (must match the trained model) ---
    const int VOCAB_SIZE = 30522;
    const int D_MODEL = 512;
    const int MAX_SEQ_LEN = 256;
    const int N_HEADS = 8;
    const int D_FF = 2048;
    const int N_LAYERS = 6; // Use 6 for both Encoder and Decoder as is standard

    // --- 1. Open the binary weights file ---
    std::ifstream weight_file("model_weights.bin", std::ios::binary);
    if (!weight_file.is_open()) {
        std::cerr << "ERROR: Could not open model_weights.bin" << std::endl;
        return 1;
    }
    std::cout << "Successfully opened model_weights.bin" << std::endl;

    // --- 2. Construct the full model by loading weights ---
    std::cout << "Constructing model with trained weights..." << std::endl;
    InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, weight_file);
    Encoder encoder(N_LAYERS, D_MODEL, N_HEADS, D_FF, weight_file);
    Decoder decoder(N_LAYERS, D_MODEL, N_HEADS, D_FF, weight_file);
    std::cout << "Model constructed successfully." << std::endl;

    // Close the file after all weights are loaded
    weight_file.close();

    // --- 3. Create sample inputs for a full forward pass ---
    std::vector<int> source_tokens = {101, 2422, 2035, 1012, 8833, 102}; // Input: "[CLS] find all . log [SEP]"
    std::vector<int> target_tokens = {101, 1000, 2035, 1000};             // Target so far: "[CLS] {\"actions\":..." (simulated)

    // --- 4. Perform the full Encoder-Decoder forward pass ---
    std::cout << "\n--- Starting Forward Pass ---" << std::endl;

    // a. The encoder processes the source sentence
    Eigen::MatrixXf source_matrix = input_layer.forward(source_tokens);
    Eigen::MatrixXf encoder_output = encoder.forward(source_matrix);
    std::cout << "Encoder output shape: " << encoder_output.rows() << "x" << encoder_output.cols() << std::endl;

    // b. The decoder processes the target sentence so far, using the encoder's output for context
    Eigen::MatrixXf target_matrix = input_layer.forward(target_tokens);
    Eigen::MatrixXf decoder_output = decoder.forward(target_matrix, encoder_output);
    std::cout << "Decoder output shape: " << decoder_output.rows() << "x" << decoder_output.cols() << std::endl;

    std::cout << "\nCongratulations! You have successfully passed data through the complete Encoder-Decoder Transformer." << std::endl;

    return 0;
}