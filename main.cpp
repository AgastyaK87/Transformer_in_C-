#include <iostream>
#include <vector>
#include "input_layer.h"
#include "attention.h"
#include "layer_norm.h"
#include "ffn.h"

int main() {
    // --- Hyperparameters ---
    const int VOCAB_SIZE = 1000;
    const int D_MODEL = 512;
    const int MAX_SEQ_LEN = 50;
    const int N_HEADS = 8;
    const int D_K = 64;
    const int D_FF = 2048; // Standard inner dimension for the FFN

    // --- Create all the layers for one complete Encoder Block ---
    InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN);
    MultiHeadAttention multi_head_attention(N_HEADS, D_MODEL, D_K);
    LayerNorm norm1(D_MODEL);
    FFN feed_forward_network(D_MODEL, D_FF);
    LayerNorm norm2(D_MODEL);

    // --- Create a sample input sentence ---
    std::vector<int> sample_tokens = {15, 234, 512, 9, 87, 34};

    // --- Simulate a full Encoder Block Forward Pass ---

    // 1. Input Layer
    Eigen::MatrixXf x = input_layer.forward(sample_tokens);
    std::cout << "1. After Input Layer, shape: " << x.rows() << "x" << x.cols() << std::endl;

    // 2. Multi-Head Attention
    Eigen::MatrixXf attn_output = multi_head_attention.forward(x);
    std::cout << "2. After Multi-Head Attention, shape: " << attn_output.rows() << "x" << attn_output.cols() << std::endl;

    // 3. First "Add & Norm"
    //    Here we add the input of the attention layer (x) to its output.
    x = norm1.forward(x + attn_output);
    std::cout << "3. After first Add & Norm, shape: " << x.rows() << "x" << x.cols() << std::endl;

    // 4. Feed-Forward Network
    Eigen::MatrixXf ffn_output = feed_forward_network.forward(x);
    std::cout << "4. After FFN, shape: " << ffn_output.rows() << "x" << ffn_output.cols() << std::endl;

    // 5. Second "Add & Norm"
    //    Here we add the input of the FFN (x) to its output.
    x = norm2.forward(x + ffn_output);
    std::cout << "5. After second Add & Norm (Final Output), shape: " << x.rows() << "x" << x.cols() << std::endl;

    std::cout << "\nSuccessfully passed data through a full Transformer Encoder block." << std::endl;

    return 0;
}