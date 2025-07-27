#ifndef DECODER_H
#define DECODER_H

#include "attention.h"
#include "layer_norm.h"
#include "ffn.h"
#include <Eigen/Dense>
#include <vector>
#include <fstream>


class DecoderBlock
{
public:
    DecoderBlock(int d_model, int n_heads, int d_ff, std::ifstream& weight_file);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x, const Eigen::MatrixXf& encoder_output);

private:
    //2 attn layers
    MultiHeadAttention masked_self_attention_;
    MultiHeadAttention cross_attention_;
    FFN ffn_;

    LayerNorm norm1_;
    LayerNorm norm2_;
    LayerNorm norm3_;


};


class Decoder
{
public:
    Decoder(int n_layers, int d_model, int n_heads, int d_ff, std::ifstream& weight_file);

    //Forward Pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x, const Eigen::MatrixXf& encoder_output);
private:
    std::vector<DecoderBlock> layers_;
};
#endif //DECODER_H
