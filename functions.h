#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

namespace functions {
    VectorXd sigmoid(const VectorXd &v) {
        return 1.0 / (1.0 + (-v.array()).exp());
    }

    VectorXd sigmoid_derivative(const VectorXd &v) {
        return sigmoid(v).array() * (1.0 - sigmoid(v).array());
    }

    double error_function(const VectorXd &output, const VectorXd &target) {
        return 0.5 * (output - target).squaredNorm();
    }

    VectorXd error_function_derivative(const VectorXd &output, const VectorXd &target) {
        return output - target;
    }
}

#endif // FUNCTIONS_H