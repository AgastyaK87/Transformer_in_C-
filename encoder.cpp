#include "encoder.h"

EncoderBlock::EncoderBlock(int d_model, int n_heads, int d_ff) :
    d_model_(d_model),
    attention_(n_heads, d_model, d_model / n_heads), // d_k is typically d_model / n_heads
    norm1_(d_model),
    ffn_(d_model, d_ff),
    norm2_(d_model)
{}

Eigen::MatrixXf EncoderBlock::forward(const Eigen::MatrixXf& x) {
    // 1. Multi-Head Attention sub-layer
    Eigen::MatrixXf attn_output = attention_.forward(x);

    // 2. First Add & Norm
    //    The input to the attention layer (x) is added to its output.
    Eigen::MatrixXf norm1_output = norm1_.forward(x + attn_output);

    // 3. Feed-Forward Network sub-layer
    Eigen::MatrixXf ffn_output = ffn_.forward(norm1_output);

    // 4. Second Add & Norm
    //    The input to the FFN (norm1_output) is added to its output.
    Eigen::MatrixXf final_output = norm2_.forward(norm1_output + ffn_output);

    return final_output;
}

Encoder::Encoder(int n_layers, int d_model, int n_heads, int d_ff) {
    // Create the stack of N identical encoder blocks
    for (int i = 0; i < n_layers; ++i) {
        layers_.emplace_back(d_model, n_heads, d_ff);
    }
}

Eigen::MatrixXf Encoder::forward(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf current_x = x;

    // Pass the input through each layer in the stack sequentially
    for (auto& layer : layers_) {
        current_x = layer.forward(current_x);
    }

    return current_x;
}