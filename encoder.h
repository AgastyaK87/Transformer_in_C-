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

#endif //ENCODER_H