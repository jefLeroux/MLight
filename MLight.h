#ifndef MLIGHT_H
#define MLIGHT_H

#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifndef MLIGHT_ACT
#define MLIGHT_ACT ACT_SIG
#endif

#ifndef MLIGHT_RELU_PARAM
#define MLIGHT_RELU_PARAM 0.01f
#endif

#ifndef MLIGHT_MALLOC
#include <stdlib.h>
#define MLIGHT_MALLOC malloc
#endif

#ifndef MLIGHT_ASSERT
#include <assert.h>
#define MLIGHT_ASSERT assert
#endif 

#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

typedef enum {
    ACT_SIG,
    ACT_RELU,
    ACT_TANH,
    ACT_SIN,
} Act;

float rand_float(void);

float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);
float sinf(float x);

float actf(float x, Act act);
float dactf(float y, Act act);

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

Mat matrix_allocate(size_t rows, size_t cols);
void matrix_fill(Mat m, float value);
void matrix_save(FILE *out, Mat m);
Mat matrix_load(FILE *in);
void matrix_randomize(Mat m, float low, float high);
Mat matrix_row(Mat m, size_t row);
void matrix_copy(Mat dst, Mat src);
void matrix_multiply(Mat dst, Mat a, Mat b);
void matrix_sum(Mat dst, Mat a);
void matrix_activate(Mat m);
void matrix_print(Mat m, const char *name, size_t padding);
void matrix_shuffle_rows(Mat m);

#define MATRIX_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count + 1
} NN;

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
void nn_randomize(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat tin, Mat tout);
void nn_finite_diff(NN nn, NN g, float eps, Mat tin, Mat tout);
void nn_backprop(NN nn, NN g, Mat tin, Mat tout);
void nn_learn(NN nn, NN g, float rate);

#define NEURALNETWORK_PRINT(nn) nn_print(nn, #nn)
#define NEURALNETWORK_RANDOMIZE(nn) nn_randomize(nn, 0, 1)

#define NEURALNETWORK_INPUT(nn) (nn).as[0]
#define NEURALNETWORK_OUTPUT(nn) (nn).as[(nn).count]

typedef struct {
    size_t begin;
    float cost;
    bool finished;
} Batch;

void batch_process(Batch *batch, size_t batch_size, NN nn, NN g, Mat t, float rate);

#ifdef MLIGHT_ENABLE_INSIGHTML
#include "raylib.h"
#include <float.h>

typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} Plot;

#define DA_INIT_CAP 256
#define da_append(da, item)                                                          \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)

void insightML_render(NN nn, float rx, float ry, float rw, float rh);
void insightML_plot(Plot p, float rx, float ry, float rw, float rh);

#endif // MLIGHT_ENABLE_INSIGHTML

#endif // MLIGHT_H

#ifdef MLIGHT_IMPLEMENTATION

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
 * The ReLU (Rectified Linear Unit) function with a parameter for negative values.
 * 
 * This function returns the input value if it is greater than zero; otherwise,
 * it returns the input value multiplied by the constant MLIGHT_RELU_PARAM.
 * 
 * @param x The input value to the ReLU function.
 * 
 * @return The output of the ReLU function, either the input value or the input 
 *         value scaled by MLIGHT_RELU_PARAM if the input is non-positive.
 */
float reluf(float x) { 
    return x > 0 ? x : x*MLIGHT_RELU_PARAM; 
}

/**
 * The hyperbolic tangent function maps any real-valued number to a value between -1 and 1.
 * 
 * This function returns (exp(x) - exp(-x)) / (exp(x) + exp(-x)) for the input x.
 * 
 * @param x The input to the hyperbolic tangent function.
 * 
 * @return The output of the hyperbolic tangent function.
 */
float thanhf(float x) {
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x)); 
}

/**
 * Computes the sine of the given angle (in radians).
 * 
 * This function calculates the sine of the input angle using the standard
 * library sine function and returns the result as a float.
 * 
 * @param x The angle in radians for which to compute the sine.
 * 
 * @return The sine of the given angle.
 */
float sinf(float x) {
    return (float) sin(x);
}

/**
 * Applies an activation function to the given value.
 *
 * @param x The input value to the activation function.
 * @param act The type of activation function to apply.
 *
 * @return The output of the activation function.
 */
float actf(float x, Act act)
{
    switch (act) {
    case ACT_SIG:  return sigmoidf(x);
    case ACT_RELU: return reluf(x);
    case ACT_TANH: return tanhf(x);
    case ACT_SIN:  return sinf(x);
    }
    MLIGHT_ASSERT(0 && "Unreachable");
    return 0.0f;
}

/**
 * Computes the derivative of the activation function for a given output value.
 *
 * This function calculates the derivative of the specified activation function
 * based on the output value `y` and the type of activation function `act`. 
 * The derivative is used in backpropagation during neural network training.
 *
 * @param y The output value of the activation function.
 * @param act The type of activation function used (e.g., sigmoid, ReLU, tanh, sine).
 *
 * @return The derivative of the activation function with respect to its input.
 */
float dactf(float y, Act act)
{
    switch (act) {
    case ACT_SIG:  return y*(1 - y);
    case ACT_RELU: return y >= 0 ? 1 : MLIGHT_RELU_PARAM;
    case ACT_TANH: return 1 - y*y;
    case ACT_SIN:  return cosf(asinf(y));
    }
    MLIGHT_ASSERT(0 && "Unreachable");
    return 0.0f;
}

/**
 * Fills all elements of a matrix with the specified value.
 *
 * @param m The matrix to be filled.
 * @param value The value to fill the matrix with.
 */
void matrix_fill(Mat m, float value) {
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
 * @return A matrix structure with allocated memory for elements.
 *         The 'stride' is set to the number of columns.
 *         An assertion ensures that memory allocation is successful.
 */
Mat matrix_allocate(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(sizeof(*m.es) * rows * cols);
    MLIGHT_ASSERT(m.es != NULL);
    return m;
};

/**
 * Saves a matrix to a binary file.
 * 
 * This function writes a magic identifier followed by the matrix's 
 * dimensions and elements to a binary file. The data is written in the 
 * following order: a magic string, the number of rows, the number of 
 * columns, and the matrix elements.
 * 
 * @param out The file pointer where the matrix will be saved. 
 *            It should be opened in binary write mode.
 * @param m The matrix to save. The matrix's elements are written 
 *          consecutively in row-major order.
 */
void matrix_save(FILE *out, Mat m) {
    const char *magic = "MLight.M";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for(size_t i = 0; i < m.rows; i++) {
        size_t n = fwrite(&MATRIX_AT(m, i, 0), sizeof(*m.es), m.cols, out);
        while(n < m.cols && !ferror(out)) {
            size_t k = fwrite(m.es + n, sizeof(*m.es), m.cols - n, out);
            n += k;
        }
    }
}

/**
 * Loads a matrix from a binary file.
 * 
 * This function reads a magic identifier followed by the matrix's 
 * dimensions and elements from a binary file. The data is read in the 
 * following order: a magic string, the number of rows, the number of 
 * columns, and the matrix elements.
 * 
 * @param in The file pointer where the matrix is saved. It should be 
 *            opened in binary read mode.
 * @return The loaded matrix.
 */
Mat matrix_load(FILE *in) { 
    uint64_t magic;
    fread(&magic, sizeof(magic), 1, in);
    // can be magic needs to be reversed

    MLIGHT_ASSERT(magic == 0x4d2e746867694c4d);
    size_t rows, cols;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);
    Mat m = matrix_allocate(rows, cols);
    
    size_t n = fread(m.es, sizeof(*m.es), rows * cols, in);
    while(n < rows * cols && !ferror(in)) {
        size_t k = fread(m.es + n, sizeof(*m.es), rows * cols - n, in);
        n += k;
    }

    return m;
}

/**
 *  Fill a matrix with random values between low and high, inclusive.
 * 
 * @param m The matrix to fill
 * @param low The lowest possible value
 * @param high The highest possible value
 * 
 * @pre high > low
 * */
void matrix_randomize(Mat m, float low, float high) {
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
void matrix_multiply(Mat dst, Mat a, Mat b) {
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
Mat matrix_row(Mat m, size_t row) {
    MLIGHT_ASSERT(row < m.rows);
    return (Mat) {
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
void matrix_copy(Mat dst, Mat src) {
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
void matrix_sum(Mat dst, Mat a) {
    MLIGHT_ASSERT(dst.rows == a.rows && dst.cols == a.cols);
    for(size_t i = 0; i < a.rows; i++) {
        for(size_t j = 0; j < a.cols; j++) {
            MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
        }   
    }
};

/**
 * Apply an activation function to each element of a matrix
 *
 * \param m The matrix to apply the sigmoid function to.
 */
void matrix_activate(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MATRIX_AT(m, i, j) = actf(MATRIX_AT(m, i, j), MLIGHT_ACT);
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
void matrix_print(Mat m, const char *name, size_t padding) {
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
 * Randomly shuffles the rows of a matrix in-place.
 * 
 * Uses the Fisher-Yates shuffle algorithm, which runs in O(n) time.
 * 
 * @param m The matrix to shuffle
 */
void matrix_shuffle_rows(Mat m) {
    for (size_t i = 0; i < m.rows; i++) {
        size_t j = (i + rand()) % (m.rows - i);
        if ( i != j ) {
            for(size_t k = 0; k < m.cols; k++) {
                float tmp = MATRIX_AT(m, i, k);
                MATRIX_AT(m, i, k) = MATRIX_AT(m, j, k);
                MATRIX_AT(m, j, k) = tmp;
            }
        }
    }
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
        matrix_activate(nn.as[i + 1]);
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
float nn_cost(NN nn, Mat tin, Mat tout) {
    MLIGHT_ASSERT(tin.rows == tout.rows);  
    MLIGHT_ASSERT(tout.cols == NEURALNETWORK_OUTPUT(nn).cols);
    size_t rows = tin.rows;

    float cost = 0;
    for (size_t i = 0; i < rows; i++) {
        Mat x = matrix_row(tin, i);
        Mat y = matrix_row(tout, i);

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
void nn_finite_diff(NN nn, NN g, float eps, Mat tin, Mat tout) {
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
void nn_backprop(NN nn, NN g, Mat tin, Mat tout) {
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
            MATRIX_AT(NEURALNETWORK_OUTPUT(g), 0, j) = 2 * (MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, j) - MATRIX_AT(tout, i, j));
        }
            float s = 1;


        for(size_t l = nn.count; l > 0; l--) {
            for(size_t j = 0; j < nn.as[l].cols; j++) {
                float a = MATRIX_AT(nn.as[l], 0, j);
                float da = MATRIX_AT(g.as[l], 0, j);
                float qa = dactf(a, MLIGHT_ACT);
                MATRIX_AT(g.bs[l-1], 0, j) += s*da*qa;
                for (size_t k = 0; k < nn.as[l - 1].cols; k++) {
                    // j weigth matrix col
                    // k weight matrix row
                    float pa = MATRIX_AT(nn.as[l - 1], 0, k);
                    float w = MATRIX_AT(nn.ws[l - 1], k, j);
                    MATRIX_AT(g.ws[l - 1], k, j) += s*da*qa * pa;
                    MATRIX_AT(g.as[l - 1], 0, k) += s*da*qa * w;
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

/**
 * Processes a batch of the given training data. 
 * By splitting the data into batches, we can more efficiently train the neural network with more data.
 *
 * @param batch The batch to process. The batch is modified.
 * @param batch_size The size of the batch.
 * @param nn The neural network to process the batch with.
 * @param g The gradients of the neural network's weights and biases.
 * @param t The training data to process.
 * @param rate The learning rate.
 * 
 * @pre batch -> begin < t.rows
 * @pre batch -> finished == false
 * @post batch -> finished == (batch -> begin >= t.rows)
 * @post batch -> cost is the average cost of the batch
 */
void batch_process(Batch *batch, size_t batch_size, NN nn, NN g, Mat t, float rate) {
    if (batch -> finished) {
        batch -> finished = false;
        batch -> begin = 0;
        batch -> cost = 0;
    }

    size_t size = batch_size;
    if( batch -> begin + batch_size >= t.rows) {
        size = t.rows - batch -> begin;
    }

    Mat batch_tin = {
        .rows = size,
        .cols = NEURALNETWORK_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MATRIX_AT(t, batch -> begin, 0)   
    };

    Mat batch_tout = {
        .rows = size,
        .cols = NEURALNETWORK_OUTPUT(nn).cols,
        .stride = t.stride,
        .es = &MATRIX_AT(t, batch -> begin, batch_tin.cols)   
    };

    nn_backprop(nn, g, batch_tin, batch_tout);
    nn_learn(nn, g, rate);

    batch -> cost += nn_cost(nn, batch_tin, batch_tout);
    batch -> begin += size;

    if (batch -> begin >= t.rows) {
        size_t batch_count = (t.rows + batch_size - 1) / batch_size;
        batch -> cost /= batch_count;
        batch -> finished = true;
    }
}


#ifdef MLIGHT_ENABLE_INSIGHTML

/**
 * Renders the neural network as a graph.
 *
 * This function renders the neural network as a graph with neurons as circles
 * and weights as lines. The color of each weight is determined by the
 * corresponding weight value, and the color of each neuron is determined by
 * the corresponding bias value.
 *
 * @param nn The neural network to render.
 * @param rx The x-coordinate of the top left corner of the neural network's
 *            bounding box.
 * @param ry The y-coordinate of the top left corner of the neural network's
 *            bounding box.
 * @param rw The width of the neural network's bounding box.
 * @param rh The height of the neural network's bounding box.
 */
void insightML_render(NN nn, float rx, float ry, float rw, float rh) {
    Color low_color = BLUE;
    Color high_color = ORANGE;

    float neuron_radius = rh * 0.03f;
    float layer_border_vpad = rh*0.08f;
    float layer_border_hpad = rw*0.06;
    float nn_width = rw - 2*layer_border_hpad;
    float nn_height = rh - 2*layer_border_vpad;
    float nn_x = rx + rw/2 - nn_width/2;
    float nn_y = ry + rh/ 2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    float layer_hpad = nn_width / arch_count;
    for(size_t l = 0; l < arch_count; l++) {
        float layer_vpad1 = nn_height / nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            float cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            float cy1 = nn_y + i*layer_vpad1  + layer_vpad1/2;
            if (l + 1 < arch_count) {
                float layer_vpad2 = nn_height / nn.as[l + 1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    float cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2; 
                    float cy2 = nn_y + j*layer_vpad2  + layer_vpad2/2;
                    float value = sigmoidf(MATRIX_AT(nn.ws[l], j, i));
                    high_color.a = floorf(255.0f * value);
                    float thick = rh * 0.003f;
                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.0f * sigmoidf(MATRIX_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

/**
 * Renders a plot of the given data points.
 *
 * This function renders a line graph of the given data points. The x-axis is
 * scaled to fit all the data points, and the y-axis is scaled to fit the
 * minimum and maximum values of the data points. The graph is drawn in red, and
 * the x-axis is drawn in white. The y-axis is not drawn, but the y-coordinate
 * of the x-axis is labeled with the value "0".
 *
 * @param plot The data points to render.
 * @param rx The x-coordinate of the top left corner of the bounding box of the
 *            graph.
 * @param ry The y-coordinate of the top left corner of the bounding box of the
 *            graph.
 * @param rw The width of the bounding box of the graph.
 * @param rh The height of the bounding box of the graph.
 */
void insightML_plot(Plot plot, float rx, float ry, float rw, float rh) {
    float min = FLT_MAX, max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (max < plot.items[i]) max = plot.items[i];
        if (min > plot.items[i]) min = plot.items[i];
    }

    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 1000) n = 1000;
    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = rx + (float)rw/n*i;
        float y1 = ry + (1 - (plot.items[i] - min)/(max - min))*rh;
        float x2 = rx + (float)rw/n*(i+1);
        float y2 = ry + (1 - (plot.items[i+1] - min)/(max - min))*rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005, RED);
    }

    float y0 = ry + (1 - (0 - min)/(max - min))*rh;
    DrawLineEx((Vector2){rx + 0, y0}, (Vector2){rx + rw - 1, y0}, rh*0.005, WHITE);
    DrawText("0", rx + 0, y0 - rh*0.04, rh*0.04, WHITE);
}

#endif // MLIGHT_ENABLE_INSIGHTML

#endif // MLIGHT_IMPLEMENTATION