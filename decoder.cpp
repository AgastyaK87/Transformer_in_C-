#include "decoder.h"


DecoderBlock::DecoderBlock(int d_model, int n_heads, int d_ff, std::ifstream& weight_file) :
    // Initialize all the sub-layers, loading weights for each one
    masked_self_attention_(n_heads, d_model, d_model / n_heads, weight_file),
    norm1_(d_model, weight_file),
    cross_attention_(n_heads, d_model, d_model / n_heads, weight_file),
    norm2_(d_model, weight_file),
    ffn_(d_model, d_ff, weight_file),
    norm3_(d_model, weight_file)
{}


Eigen::MatrixXf DecoderBlock::forward(const Eigen::MatrixXf& x, const Eigen::MatrixXf& encoder_output)
{
    //Masked Self attn
    //decoder attentions the output it generated here, mask is applied to prevent peeking ahead
    Eigen::MatrixXf self_attn_output = masked_self_attention_.forward(x);

    //2. add and norm

    Eigen::MatrixXf norm1_output = norm1_.forward(x + self_attn_output);

    //Cross Attention
    //Decoder looks at encoder output
    //Query comes from decoders state (norm1 output), Key and Value come from encoder final output
    //FUTURE: MODIFY THE MULTIHEAD ATTN CLASS TO ACCEPT SEPARATE K AND V INPUTS
    // PASS NORM1 OUTPUT TO STANDARD ATTN BLOCK - FIX THIS LATER.
    // IN TH EFUTURE: multihead attn forward method signature needs to accept an optional encoder.


    Eigen::MatrixXf cross_attn_output = cross_attention_.forward(norm1_output);

    Eigen::MatrixXf norm2_output = norm2_.forward(norm1_output + cross_attn_output);

    Eigen::MatrixXf ffn_output = ffn_.forward(norm2_output);

    Eigen::MatrixXf final_output = norm3_.forward(norm2_output + ffn_output);

    return final_output;
}


Decoder::Decoder(int n_layers, int d_model, int n_heads, int d_ff, std::ifstream& weight_file )
{
    for (int i = 0; i < n_layers; i++)
    {
        layers_.emplace_back(d_model, n_heads, d_ff, weight_file);
    }
}

Eigen::MatrixXf Decoder::forward(const Eigen::MatrixXf& x, const Eigen::MatrixXf& encoder_output)
{
    Eigen::MatrixXf current_x = x;

    for (auto& layer: layers_)
    {
        current_x = layer.forward(current_x, encoder_output);
    }
    return current_x;
}