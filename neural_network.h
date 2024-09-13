#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <Eigen/Dense>
#include <functional>

#include "layer.h"
#include "functions.h"


using namespace Eigen;

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    
public:
    NeuralNetwork(std::vector<Layer> layers) : layers(layers) {}


    void forward(VectorXd& input) {
        layers[0].forward(input);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].forward(layers[i - 1].get_neurons_values_activate());
        }
    }

    VectorXd get_output() {
        return layers.back().get_neurons_values_activate();
    }

    void backward(VectorXd& input, VectorXd& target_output, double learning_rate) {
        VectorXd output_error = functions::error_function_derivative(get_output(), target_output);
        VectorXd output_derivative = layers.back().derivative_of_activation_function();
        VectorXd output_delta = output_error.cwiseProduct(output_derivative);
        layers.back().set_delta(output_delta);

        for (int i = layers.size() - 2; i >= 0; i--) {
            MatrixXd next_weights = layers[i + 1].get_weights();
            VectorXd next_delta = layers[i + 1].get_delta();

            VectorXd hidden_error = next_weights * next_delta;
            VectorXd hidden_derivative = layers[i].derivative_of_activation_function();
            VectorXd hidden_delta = hidden_error.cwiseProduct(hidden_derivative);
            layers[i].set_delta(hidden_delta);
        }

        for (int i = 0; i < layers.size(); i++) {
            VectorXd input_;
            if (i == 0) {
                input_ = input;
            } 
            else {
                input_ = layers[i - 1].get_neurons_values_activate();
            }

            layers[i].update_weights(input_, learning_rate);
        }
    }

    void train(const std::vector<VectorXd>& inputs, 
                const std::vector<VectorXd>& target_outputs, 
                double learning_rate, int epochs) {

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            
            for (size_t i = 0; i < inputs.size(); ++i) {
                VectorXd input = inputs[i];
                VectorXd target_output = target_outputs[i];

                forward(input);

                VectorXd output = get_output();
                double loss = functions::error_function(output, target_output);
                epoch_loss += loss;

                backward(input, target_output, learning_rate);
            }
            epoch_loss /= inputs.size();
            
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
        }
    }

    void test(const std::vector<VectorXd>& inputs, 
            const std::vector<VectorXd>& target_outputs) {

        int correct_predictions = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            VectorXd input = inputs[i];
            VectorXd target_output = target_outputs[i];

            forward(input);

            VectorXd output = get_output();

            int predicted_index;
            int target_index;
            output.maxCoeff(&predicted_index);
            target_output.maxCoeff(&target_index);

            if (predicted_index == target_index) {
                correct_predictions++;
            }
        }

        double accuracy = static_cast<double>(correct_predictions) / inputs.size();
        std::cout << "Test Accuracy: " << accuracy << std::endl;
    }
};

#endif // NEURAL_NETWORK_H