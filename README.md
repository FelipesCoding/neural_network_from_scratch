# Neural Network from Scratch

## Description

This project implements a basic feedforward neural network from scratch in C++ using the Eigen library for efficient matrix operations. The network is designed to demonstrate core concepts of neural networks such as forward propagation, backpropagation, and gradient descent without relying on high-level machine learning frameworks. It is a great learning resource to understand the inner workings of neural networks and the power of matrix operations in deep learning.

## Prerequisites

- **C++ Compiler**: Ensure you have a modern C++ compiler that supports **C++20** (required for certain functionalities such as `ntohl` for endianess conversion).
- **Eigen Library**: This project uses the Eigen library for matrix operations. [Download Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) if it's not already installed on your system.

### Configuring Eigen

The project expects Eigen to be located at `/opt/homebrew/include/eigen3`. If Eigen is installed in a different directory on your system, you need to update the `Makefile` accordingly:

1. Open the `Makefile`.
2. Locate the line that sets the `CXXFLAGS` variable:
    ```makefile
    CXXFLAGS = -I /opt/homebrew/include/eigen3
    ```
3. Update the path to match the location where Eigen is installed on your system. For example, if Eigen is installed in `/usr/local/include/eigen3`, you should modify the line to:
    ```makefile
    CXXFLAGS = -I /usr/local/include/eigen3
    ```

## Build Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/FelipesCoding/neural_network_from_scratch.git
    cd neural_network_from_scratch
    ```

2. **Update the Eigen path** in the `Makefile` if necessary.

3. **Build the project** using `make`:
    ```bash
    make
    ```

4. **Run the executable**:
    ```bash
    ./main
    ```

## Dataset

The MNIST dataset can be downloaded from [Yann LeCun's website](https://yann.lecun.com/exdb/mnist/). You will need the following files:

- **Training images**: `train-images-idx3-ubyte`
- **Training labels**: `train-labels-idx1-ubyte`
- **Test images**: `t10k-images-idx3-ubyte`
- **Test labels**: `t10k-labels-idx1-ubyte`

After downloading, place these files in the `data/MNIST/raw/` directory of your project.

## Example Usage

Here’s a basic example demonstrating how to use the neural network:

```cpp
#include <iostream>
#include <Eigen/Dense>

#include "layer.h"
#include "functions.h"
#include "neural_network.h"
#include "utils.h"

using namespace Eigen;
using namespace functions;

// File paths to the MNIST dataset
const std::string mnist_train_data_path = "data/MNIST/raw/train-images-idx3-ubyte";
const std::string mnist_train_label_path = "data/MNIST/raw/train-labels-idx1-ubyte";
const std::string mnist_test_data_path = "data/MNIST/raw/t10k-images-idx3-ubyte";
const std::string mnist_test_label_path = "data/MNIST/raw/t10k-labels-idx1-ubyte";

int main() {
    srand(time(0));

    // Prepare datasets
    std::vector<VectorXd> train_dataset;
    std::vector<VectorXd> label_train_dataset;
    std::vector<VectorXd> test_dataset;
    std::vector<VectorXd> label_test_dataset;

    // Read MNIST data
    utils::read_mnist_train_data(mnist_train_data_path, train_dataset);
    utils::read_mnist_train_label(mnist_train_label_path, label_train_dataset);
    utils::read_mnist_test_data(mnist_test_data_path, test_dataset);
    utils::read_mnist_test_label(mnist_test_label_path, label_test_dataset);

    // Create the neural network with 2 layers: one hidden layer and one output layer
    Layer hidden_layer(784, 64, sigmoid, sigmoid_derivative);
    Layer output_layer(64, 10, sigmoid, sigmoid_derivative);
    NeuralNetwork nn({hidden_layer, output_layer});

    // Train the network on the training dataset
    nn.train(train_dataset, label_train_dataset, 0.1, 3);

    // Test the network on the test dataset
    nn.test(test_dataset, label_test_dataset);

    return 0;
}
