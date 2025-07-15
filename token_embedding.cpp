#include "token_embedding.h"

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model) {
    // Initialize the embedding matrix to the correct size
    // and fill it with small random numbers.
    embedding_matrix_ = Eigen::MatrixXf::Random(vocab_size, d_model);
}

// This is the main lookup function.
Eigen::RowVectorXf TokenEmbedding::get_embedding(int token_id) {
    // Use Eigen's .row() method to grab the specific row
    // corresponding to the token's ID.
    return embedding_matrix_.row(token_id);
}