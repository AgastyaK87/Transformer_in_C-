
#include <fstream>
#include "utils.h" // Or the relevant header

void load_matrix(Eigen::MatrixXf& matrix, std::ifstream& file) {
    long long num_floats = matrix.rows() * matrix.cols();
    long long num_bytes = num_floats * sizeof(float);
    file.read(reinterpret_cast<char*>(matrix.data()), num_bytes);
}

void load_matrix(Eigen::RowVectorXf& vec, std::ifstream& file) {
    long long num_floats = vec.size();
    long long num_bytes = num_floats * sizeof(float);
    file.read(reinterpret_cast<char*>(vec.data()), num_bytes);
}