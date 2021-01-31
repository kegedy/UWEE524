#include "device_launch_parameters.h"
#define N 1024
#define M 1024

extern "C"
__global__ void matrix_add(float* a, float* b, float* c) {

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    // For MxN matrix, 
    // Row-Major: row*N + col
    // Col-Major: col*M + row [not implemented]
    if (row < M && col < N) {
        int tid = row * N + col;
        c[tid] = a[tid] + b[tid];
    }
}

