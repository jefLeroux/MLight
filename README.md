# MLight - Lightweight Machine Learning Library

MLight is a compact, C-based machine learning library designed for efficient neural network operations. With an intuitive API, MLight provides the essential tools for building, training, and evaluating neural networks with minimal dependencies. This library is ideal for lightweight applications, embedded systems, or projects requiring optimized, low-level control over neural networks.

## Features

- Matrix operations for fast computations
- Neural network allocation, randomization, and forward propagation
- Cost function calculation for supervised learning
- Finite difference and backpropagation for network training
- Fully documented API for easy integration

## Installation

1. Clone the repository:    
    `git clone https://github.com/username/MLight.git cd MLight`
    
2. Include `mlight.h` in your C project and compile with your project files:    
    `#include "mlight.h"`
    
3. Ensure that `malloc`, `assert`, and any standard C library dependencies are available in your environment.
    
## Quick Start

This is some example code for modeling an OR_Gate

```
int main(void) {
    // training input
    Matrix tin = {
        0, 0
        0, 1
        1, 0
        1, 1
    };
    
    // training output
    Matrix tout = {
        0,
        1,
        1,
        1
    };
    
    float rate = 1;
    bool tracing = true;
    bool network = false;
    
    size_t arch[] = {2, 2, 1}; // input, { hidden }, output
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    NEURALNETWORK_RANDOMIZE(nn);
    
    for (size_t i = 0; i < 5 *1000; i++) {
        nn_backprop(nn, g, tin, tout);
        nn_learn(nn, g, rate);
        tracing && printf("%zu: cost = %f\n", i, nn_cost(nn, tin, tout));
    }
    
    if (network) {
        NEURALNETWORK_PRINT(nn);
    }
    
    // Verify the trainned model
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MATRIX_AT(NEURALNETWORK_INPUT(nn), 0, 0) = i;
            MATRIX_AT(NEURALNETWORK_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n",i, j, MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, 0));
        }
    }
    return 0;
}
```
