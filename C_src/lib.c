// ia_step.c
#include <math.h>

#define INPUT_N 7
#define H1_N 9
#define H2_N 9
#define OUTPUT_N 4

static inline double activation(double x) {
    return 2.0 / (1.0 + exp(-x)) - 1.0;
}

/*
inputs              : double[7]
input_w             : double[7][9]
h1_w                : double[9][9]
h1_b                : double[9]
h2_w                : double[9][4]
h2_b                : double[9]
out_b               : double[4]
output              : double[4]
*/

void IA_step(
    const double *inputs,
    const double *input_w,
    const double *h1_w,
    const double *h1_b,
    const double *h2_w,
    const double *h2_b,
    const double *out_b,
    double *output
) {
    double h1_s[H1_N] = {0.0};
    double h2_s[H2_N] = {0.0};

    // hidden layer 1
    for (int i = 0; i < H1_N; i++) {
        double acc = h1_b[i];
        for (int j = 0; j < INPUT_N; j++) {
            acc += inputs[j] * input_w[j * H1_N + i];
        }
        h1_s[i] = activation(acc);
    }

    // hidden layer 2
    for (int i = 0; i < H2_N; i++) {
        double acc = h2_b[i];
        for (int j = 0; j < H1_N; j++) {
            acc += h1_s[j] * h1_w[j * H2_N + i];
        }
        h2_s[i] = activation(acc);
    }

    // output layer
    for (int i = 0; i < OUTPUT_N; i++) {
        double acc = out_b[i];
        for (int j = 0; j < H2_N; j++) {
            acc += h2_s[j] * h2_w[j * OUTPUT_N + i];
        }
        output[i] = activation(acc);
    }
}
