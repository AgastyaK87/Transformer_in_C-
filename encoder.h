#ifndef ENCODER_H
#define ENCODER_H

#include "attention.h"
#include "layer_norm.h"
#include "ffn.h"
#include <Eigen/Dense>

class EncoderBlock {
public:
    // Constructor
    EncoderBlock(int d_model, int n_heads, int d_ff);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    int d_model_;

    // Sub-layers
    MultiHeadAttention attention_;
    LayerNorm norm1_;
    FFN ffn_;
    LayerNorm norm2_;
};

// --- NEW: The final Encoder class ---
class Encoder {
public:
    // Constructor
    Encoder(int n_layers, int d_model, int n_heads, int d_ff);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    // A vector to hold the entire stack of EncoderBlocks
    std::vector<EncoderBlock> layers_;
};



#endif //ENCODER_H