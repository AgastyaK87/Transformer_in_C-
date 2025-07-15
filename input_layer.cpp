#include "input_layer.h"
#include <cmath>
#include <iostream>

InputLayer::InputLayer(int vocab_size, int d_model, int max_seq_len) : d_model_(d_model) {
    // 1. Initialize the token embedding matrix with random values
    embedding_matrix_ = Eigen::MatrixXf::Random(vocab_size, d_model);

    // 2. Create and initialize the positional encoding matrix
    pos_encoding_matrix_ = Eigen::MatrixXf::Zero(max_seq_len, d_model);
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < d_model / 2; ++i) {
            double angle = pos / std::pow(10000.0, (2.0 * i) / d_model);
            pos_encoding_matrix_(pos, 2 * i) = std::sin(angle);
            pos_encoding_matrix_(pos, 2 * i + 1) = std::cos(angle);
        }
    }
    std::cout << "InputLayer initialized." << std::endl;
}

Eigen::MatrixXf InputLayer::forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    Eigen::MatrixXf output(seq_len, d_model_);

    // For each token in the input sequence...
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];

        // 1. Get the token's embedding vector
        Eigen::RowVectorXf token_embedding = embedding_matrix_.row(token_id);

        // 2. Get the token's positional encoding vector
        Eigen::RowVectorXf pos_encoding = pos_encoding_matrix_.row(i);

        // 3. Add them together and place in the output matrix
        output.row(i) = token_embedding + pos_encoding;
    }

    return output;
}