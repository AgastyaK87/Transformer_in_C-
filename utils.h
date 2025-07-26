
#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <fstream>
void load_matrix(Eigen::MatrixXf& matrix, std::ifstream& file);

void load_matrix(Eigen::RowVectorXf& vec, std::ifstream& file);


#endif //UTILS_H
