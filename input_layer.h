#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
class InputLayer {
public:
    // Constructor to initialize the layer
    InputLayer(int vocab_size, int d_model, int max_seq_len, std::ifstream& weight_files);

    // The forward pass method
    Eigen::MatrixXf forward(const std::vector<int>& token_ids);

private:
    int d_model_;
    Eigen::MatrixXf embedding_matrix_;
    Eigen::MatrixXf pos_encoding_matrix_;
};

#endif //INPUT_LAYER_H