//
// Created by gayat on 7/14/2025.
//

#include "main.h"
#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define two 2x2 matrices
    Eigen::Matrix2f m;
    m << 1, 2,
         3, 4;

    Eigen::Matrix2f n;
    n << 5, 6,
         7, 8;

    // Multiply them
    Eigen::Matrix2f result = m * n;

    // Print the result to the console
    std::cout << "--- Eigen Library Test ---" << std::endl;
    std::cout << "\nResult of matrix multiplication:\n" << result << std::endl;

    return 0;
}