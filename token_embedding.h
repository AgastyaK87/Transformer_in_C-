#ifndef TOKEN_EMBEDDING_H
#define TOKEN_EMBEDDING_H

#include <Eigen/Dense>

class TokenEmbedding {
public:
    // Constructor: Sets up the embedding matrix
    TokenEmbedding(int vocab_size, int d_model);

    // Function to get the vector for a single word ID
    Eigen::RowVectorXf get_embedding(int token_id);

private:
    // The matrix that holds all the word vectors
    Eigen::MatrixXf embedding_matrix_;
};

#endif //TOKEN_EMBEDDING_H