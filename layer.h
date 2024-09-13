#ifndef LAYER_H
#define LAYER_H


#include <iostream>
#include <Eigen/Dense>
#include <functional>

#include "functions.h"

using namespace Eigen;

class Layer {
private:
    MatrixXd weights; 
    VectorXd biases;
    VectorXd neurons_values;
    VectorXd neurons_values_activate;
    std::function<VectorXd(const VectorXd&)> activation_function;
    std::function<VectorXd(const VectorXd&)> activation_function_derivative;
    
    VectorXd delta;

public:
    Layer(int input_size_neurons, int neurons, 
        std::function<VectorXd(const VectorXd&)> activation_function, 
        std::function<VectorXd(const VectorXd&)> activation_function_derivative) :  
            activation_function(activation_function),
            activation_function_derivative(activation_function_derivative) {
            weights = MatrixXd::Random(input_size_neurons, neurons);
            biases = VectorXd::Zero(neurons);
    }

    void forward(const VectorXd &input) {
        VectorXd Y = weights.transpose() * input + biases;
        neurons_values = Y;
        neurons_values_activate = activation_function(Y);
    }


    VectorXd get_neurons_values_activate() {
        return neurons_values_activate;
    }

    void set_delta(const VectorXd& delta) {
        this->delta = delta;
    }

    VectorXd get_delta() {
        return delta;
    }

    MatrixXd get_weights() {
        return weights;
    }

    void update_weights(const VectorXd& input, double learning_rate) {
        weights -= learning_rate * input * delta.transpose();
        biases -= learning_rate * delta;
    }

    VectorXd derivative_of_activation_function() {
        return activation_function_derivative(neurons_values);
    }
};

#endif // LAYER_H