#ifndef MLIGHT_H
#define MLIGHT_H

#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifndef MLIGHT_MALLOC
#include <stdlib.h>
#define MLIGHT_MALLOC malloc
#endif

#ifndef MLIGHT_ASSERT
#include <assert.h>
#define MLIGHT_ASSERT assert

#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Matrix;

Matrix matrix_allocate(size_t rows, size_t cols);
void matrix_fill(Matrix m, float value);
void matrix_randomize(Matrix m, float low, float high);
Matrix matrix_row(Matrix m, size_t row);
void matrix_copy(Matrix dst, Matrix src);
void matrix_multiply(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void matrix_sigmoid(Matrix m);
void matrix_print(Matrix m, const char *name, size_t padding);

#define MATRIX_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)

typedef struct {
    size_t count;
    Matrix *ws;
    Matrix *bs;
    Matrix *as; // The amount of activations is count + 1
} NN;

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
void nn_randomize(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Matrix tin, Matrix tout);
void nn_finite_diff(NN nn, NN g, float eps, Matrix tin, Matrix tout);
void nn_backprop(NN nn, NN g, Matrix tin, Matrix tout);
void nn_learn(NN nn, NN g, float rate);

#define NEURALNETWORK_PRINT(nn) nn_print(nn, #nn)
#define NEURALNETWORK_RANDOMIZE(nn) nn_randomize(nn, 0, 1)

#define NEURALNETWORK_INPUT(nn) (nn).as[0]
#define NEURALNETWORK_OUTPUT(nn) (nn).as[(nn).count]


#endif

/**
 * Generates a random floating point number between 0.0 and 1.0.
 * 
 * @return A random floating point number between 0.0 and 1.0.
 */
float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

/**
 * The sigmoid function maps any real-valued number to a value between 0 and 1.
 * 
 * @param x The input to the sigmoid function.
 * 
 * @return The output of the sigmoid function.
 */
float sigmoidf(float x) {
    return 1.0f / (1.0f + exp(-x));
}

/**
 * Fills all elements of a matrix with the specified value.
 *
 * @param m The matrix to be filled.
 * @param value The value to fill the matrix with.
 */
void matrix_fill(Matrix m, float value) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = value;
        }
    }
}

/**
 * Allocates a matrix with the specified number of rows and columns.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * 
 * @pre rows > 0
 * @pre cols > 0
 * 
 * @return A Matrix structure with allocated memory for elements.
 *         The 'stride' is set to the number of columns.
 *         An assertion ensures that memory allocation is successful.
 */
Matrix matrix_allocate(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(sizeof(*m.es) * rows * cols);
    MLIGHT_ASSERT(m.es != NULL);
    return m;
};

/**
 *  Fill a matrix with random values between low and high, inclusive.
 * 
 * @param m The matrix to fill
 * @param low The lowest possible value
 * @param high The highest possible value
 * 
 * @pre high > low
 * */
void matrix_randomize(Matrix m, float low, float high) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MLIGHT_ASSERT(high > low);
            MATRIX_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

/**
 *  Multiply two matrices and write the result to a destination matrix.
 * 
 * @param dst The destination matrix
 * @param a The first matrix to multiply
 * @param b The second matrix to multiply
 * 
 * @pre a.cols == b.rows
 * @pre dst.rows == a.rows
 * @pre dst.cols == b.cols
 * @pre dst is a valid matrix with enough space to store the result
 */
void matrix_multiply(Matrix dst, Matrix a, Matrix b) {
    MLIGHT_ASSERT(a.cols == b.rows);
    float inner_size = a.cols;
    MLIGHT_ASSERT(dst.rows == a.rows && dst.cols == b.cols);

    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t j = 0; j < dst.cols; j++) {
            float sum = 0;
            for(size_t k = 0; k < inner_size; k++) {
                sum += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
            }
            MATRIX_AT(dst, i, j) = sum;
        }
    }
};

/**
 *  Creates a new matrix from a subset of another matrix
 * 
 * @param m The matrix to view
 * @param row The row to view
 * @return A matrix with only the specified row of m
 */
Matrix matrix_row(Matrix m, size_t row) {
    MLIGHT_ASSERT(row < m.rows);
    return (Matrix) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MATRIX_AT(m, row, 0)
    };
};

/**
 * Copies the values from one matrix into another.
 * 
 * @param dst The destination matrix which will also hold the result
 * @param src The matrix to copy
 * 
 * @pre dst.rows == src.rows
 */
void matrix_copy(Matrix dst, Matrix src) {
    MLIGHT_ASSERT(dst.rows == src.rows && dst.cols == src.cols);
    for(size_t i = 0; i < src.rows; i++) {
        for(size_t j = 0; j < src.cols; j++) {
            MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
        }
    }
};

/**
 * Element-wise add two matrices
 * 
 * @param dst The destination matrix which will also hold the result
 * @param a The matrix to add to dst
 * 
 * @pre dst.rows == a.rows
 */
void matrix_sum(Matrix dst, Matrix a) {
    MLIGHT_ASSERT(dst.rows == a.rows && dst.cols == a.cols);
    for(size_t i = 0; i < a.rows; i++) {
        for(size_t j = 0; j < a.cols; j++) {
            MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
        }   
    }
};

/**
 * Apply the sigmoid function to each element of a matrix.
 *
 * \param m The matrix to apply the sigmoid function to.
 */
void matrix_sigmoid(Matrix m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
        }
    }
}

/**
 * Print a matrix to the console with a name and padding.
 *
 * @param m The matrix to print.
 * @param name The name of the matrix to print.
 * @param padding The amount of padding to add to the left of the matrix.
 */
void matrix_print(Matrix m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int)padding, "", name);
    for(size_t i = 0; i < m.rows; i++) {
        printf("%*s", (int)padding, "");
        for(size_t j = 0; j < m.cols; j++) {
            printf(" %f ", MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}


/**
 * Allocate a neural network from an array of layer sizes.
 *
 * @param arch An array of layer sizes. The first element is the input size, the
 *             last element is the output size, and the rest are the sizes of the
 *             hidden layers.
 * @param arch_count The number of elements in `arch`.
 * 
 * @pre arch_count > 0
 */
NN nn_alloc(size_t *arch, size_t arch_count) {
    MLIGHT_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = MLIGHT_MALLOC(sizeof(*nn.ws)*nn.count);
    MLIGHT_ASSERT(nn.ws != NULL);
    nn.bs = MLIGHT_MALLOC(sizeof(*nn.bs)*nn.count);
    MLIGHT_ASSERT(nn.bs != NULL);
    nn.as = MLIGHT_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    MLIGHT_ASSERT(nn.as != NULL);

    nn.as[0] = matrix_allocate(1, arch[0]);
    for (size_t i = 1; i < arch_count; i++) {
        nn.ws[i - 1] = matrix_allocate(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = matrix_allocate(1, arch[i]);
        nn.as[i] = matrix_allocate(1, arch[i]);
    }

    return nn;
}

/**
 * Prints the weights and biases of a neural network.
 *
 * This function outputs the values of the weight matrices (ws) and bias 
 * matrices (bs) for each layer in the neural network. The output is formatted 
 * with the given name as a prefix and each matrix is printed with an 
 * indentation for better readability.
 *
 * @param nn The neural network whose weights and biases are to be printed.
 * @param name The prefix name used in the printed output for identification.
 */
void nn_print(NN nn, const char *name) {
    char buffer[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++) {
        snprintf(buffer, sizeof(buffer), "ws%zu", i);
        matrix_print(nn.ws[i], buffer, 4);
        snprintf(buffer, sizeof(buffer), "bs%zu", i);
        matrix_print(nn.bs[i], buffer, 4);
    }
    printf("]\n");
}

/**
 * Randomizes the weights and biases of a neural network.
 *
 * This function iterates over each layer in the neural network and assigns
 * random values to the weights and biases matrices. The random values are
 * generated within the specified range [low, high].
 *
 * @param nn The neural network whose weights and biases are to be randomized.
 * @param low The lower bound of the randomization range.
 * @param high The upper bound of the randomization range.
 */
void nn_randomize(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; i++) {
        matrix_randomize(nn.ws[i], low, high);
        matrix_randomize(nn.bs[i], low, high);
    }
}

/**
 * Performs a forward pass through the neural network.
 *
 * This function iterates over each layer of the network, performing matrix
 * multiplication between the activation of the previous layer and the weights
 * of the current layer, adding the biases, and applying the sigmoid activation
 * function to the result. The output of this process is stored in the
 * activations of the current layer.
 *
 * @param nn The neural network to perform the forward pass on.
 */
void nn_forward(NN nn) {
    for(size_t i = 0; i < nn.count; i++) {
        matrix_multiply(nn.as[i + 1], nn.as[i], nn.ws[i]);
        matrix_sum(nn.as[i + 1], nn.bs[i]);
        matrix_sigmoid(nn.as[i + 1]);
    }
}

/**
 * Computes the cost of the neural network given the input and output training
 * data.
 *
 * This function first checks that the number of rows in the input and output
 * data is the same, and that the number of columns in the output data is the
 * same as the number of neurons in the output layer of the neural network.
 *
 * Then, for each row of the input and output data, it performs a forward pass
 * through the neural network, computes the error between the predicted and
 * actual output, and adds the error to a running total.
 *
 * Finally, it returns the average of the total error over all input rows.
 *
 * @param nn The neural network to compute the cost of.
 * @param tin The input training data.
 * @param tout The output training data.
 * 
 * @pre tin.rows == tout.rows
 * @pre tout.cols == NEURALNETWORK_OUTPUT(nn).cols
 * 
 * @return The cost of the neural network.
 */
float nn_cost(NN nn, Matrix tin, Matrix tout) {
    MLIGHT_ASSERT(tin.rows == tout.rows);  
    MLIGHT_ASSERT(tout.cols == NEURALNETWORK_OUTPUT(nn).cols);
    size_t rows = tin.rows;

    float cost = 0;
    for (size_t i = 0; i < rows; i++) {
        Matrix x = matrix_row(tin, i);
        Matrix y = matrix_row(tout, i);

        matrix_copy(NEURALNETWORK_INPUT(nn), x);
        nn_forward(nn);

        size_t cols = tout.cols;
        for (size_t j = 0; j < cols; j++) {
            float error = MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, j) - MATRIX_AT(y, 0, j);
            cost += error * error;
        }
    }
    return cost/rows;
}

/**
 * Computes the finite difference approximation of the cost of a
 * neural network.
 *
 * @param nn The neural network to compute the gradient of.
 * @param g The gradient of the neural network.
 * @param eps The perturbation amount used in the finite difference
 * approximation.
 * @param tin The input training data.
 * @param tout The output training data.
 * 
 * @pre tin.rows == tout.rows
 * @pre tout.cols == NEURALNETWORK_OUTPUT(nn).cols
 */
void nn_finite_diff(NN nn, NN g, float eps, Matrix tin, Matrix tout) {
    float saved;
    float c = nn_cost(nn, tin, tout);
    
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                saved = MATRIX_AT(nn.ws[i], j, k);
                MATRIX_AT(nn.ws[i], j, k) += eps;
                MATRIX_AT(g.ws[i], j, k) = (nn_cost(nn, tin, tout) - c) /eps; 
                MATRIX_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                saved = MATRIX_AT(nn.bs[i], j, k);
                MATRIX_AT(nn.bs[i], j, k) += eps;
                MATRIX_AT(g.bs[i], j, k) = (nn_cost(nn, tin, tout) - c) /eps; 
                MATRIX_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

/**
 * Sets all weights, biases, and activations of the neural network to zero.
 *
 * This function iterates through each layer of the neural network, setting
 * all elements of the weight matrices, bias matrices, and activation matrices
 * to zero. This effectively resets the state of the network.
 *
 * @param nn The neural network whose weights, biases, and activations are to be zeroed.
 */
void nn_zero(NN nn) {
    for (size_t i = 0; i < nn.count; i++) {
        matrix_fill(nn.ws[i], 0);
        matrix_fill(nn.bs[i], 0);
        matrix_fill(nn.as[i], 0);
    }
    matrix_fill(nn.as[nn.count], 0);
}


/**
 * Computes the gradient of the cost function with respect to the weights and biases of the neural network.
 *
 * This function performs a backward pass through the neural network, computing the gradients of the cost
 * function with respect to each of the weights and biases of the network. The gradients are stored in the
 * `g` parameter, which is a neural network with the same architecture as `nn`. The gradients of the cost
 * function with respect to the weights and biases are computed using the chain rule and the product rule of
 * differentiation.
 *
 * @param nn The neural network whose cost function is to be differentiated.
 * @param g The neural network that will store the gradients of the cost function.
 * @param tin The input training data.
 * @param tout The output training data.
 * 
 * @pre tin.rows == tout.rows
 * @pre tout.cols == NEURALNETWORK_OUTPUT(nn).cols
 */
void nn_backprop(NN nn, NN g, Matrix tin, Matrix tout) {
    MLIGHT_ASSERT(tin.rows == tout.rows);  
    MLIGHT_ASSERT(tout.cols == NEURALNETWORK_OUTPUT(nn).cols);
    size_t rows = tin.rows;

    nn_zero(g);

    // i - current sample
    // j - previous activation
    // l - current layer
    // k - current activation

    for (size_t i = 0; i < rows; i++) {
        matrix_copy(NEURALNETWORK_INPUT(nn), matrix_row(tin, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; ++j) {
            matrix_fill(g.as[j], 0);
        }

        for(size_t j = 0; j < tout.cols; j++) {
            MATRIX_AT(NEURALNETWORK_OUTPUT(g), 0, j) = MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, j) - MATRIX_AT(tout, i, j);
        }

        for(size_t l = nn.count; l > 0; l--) {
            for(size_t j = 0; j < nn.as[l].cols; j++) {
                float a = MATRIX_AT(nn.as[l], 0, j);
                float da = MATRIX_AT(g.as[l], 0, j);
                MATRIX_AT(g.bs[l - 1], 0, j) += 2*da*a*(1-a);
                for (size_t k = 0; k < nn.as[l - 1].cols; k++) {
                    // j weigth matrix col
                    // k weight matrix row
                    float pa = MATRIX_AT(nn.as[l - 1], 0, k);
                    float w = MATRIX_AT(nn.ws[l - 1], k, j);
                    MATRIX_AT(g.ws[l - 1], k, j) += 2*da*a*(1-a) * pa;
                    MATRIX_AT(g.as[l - 1], 0, k) += 2*da*a*(1-a) * w;
                }
            }
        }
    }

    for(size_t i = 0; i < g.count; i++) {
        for(size_t j = 0; j < g.ws[i].rows; j++) {
            for(size_t k = 0; k < g.ws[i].cols; k++) {
                MATRIX_AT(g.ws[i], j, k) /= rows;
            }
        }
        for(size_t j = 0; j < g.bs[i].rows; j++) {
            for(size_t k = 0; k < g.bs[i].cols; k++) {
                MATRIX_AT(g.bs[i], j, k) /= rows;
            }
        }
    }
}


/**
 * Updates the weights and biases of the neural network using the computed gradients.
 *
 * This function performs a gradient descent update on the weights and biases
 * of the neural network. For each weight and bias parameter, it subtracts the
 * product of the learning rate and the corresponding gradient from the parameter. 
 * This operation effectively updates the parameters in the direction that reduces 
 * the cost of the neural network, as indicated by the gradient.
 *
 * @param nn The neural network to update.
 * @param g The gradients of the neural network's weights and biases.
 * @param rate The learning rate, a scalar that controls the size of the update step.
 */
void nn_learn(NN nn, NN g, float rate) {
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                MATRIX_AT(nn.ws[i], j, k) -= rate * MATRIX_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                MATRIX_AT(nn.bs[i], j, k) -= rate * MATRIX_AT(g.bs[i], j, k);
            }
        }
    }
}

#endif

#ifdef MLIGHT_IMPLEMENTATION
#endif