#ifndef FFN_H
#define FFN_H

#include <Eigen/Dense>
#include <fstream>

class FFN {
public:
    // Constructor
    FFN(int d_model, int d_ff, std::ifstream& weight_file);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    // Helper function for ReLU activation
    Eigen::MatrixXf relu(const Eigen::MatrixXf& x);

    // Weights and biases for the two linear layers
    Eigen::MatrixXf W1_, W2_;
    Eigen::RowVectorXf b1_, b2_;
};

#endif //FFN_H