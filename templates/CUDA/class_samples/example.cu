#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "kernel.cuh"

__global__ void vecAdd_gridstride(float *result, float *a, float *b, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

void initWidth(float num, float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = num;
    }
}

void checkElementsAre(float target, float* vector, int N) {
    for (int i = 0; i < N; i++) {
        if (vector[i] != target) {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
        }
    }
    printf("SUCCESS! All values calculated correctly.\n");
}

// Error Handling Macro for CUDA Runtime
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {
    const int N = 2 << 24;
    size_t size = N * sizeof(float);

    float* a;
    float* b;
    float* c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    initWidth(3, a, N);
    initWidth(4, b, N);
    initWidth(0, c, N);

    size_t numberOfBlocks;
    size_t threadsPerBlock;

    numberOfBlocks = 32;
    threadsPerBlock = 1024;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;
    
    printf("EXEC CONFIG: Number blocks: %d, number threads: %d\n", numberOfBlocks, threadsPerBlock);
    vecadd_1d<<<numberOfBlocks, threadsPerBlock>>>(c,a,b,N);

    cudaError_t cuErrSync  = cudaGetLastError();
    if (cuErrSync != cudaSuccess) printf("sync error: %s\n", cudaGetErrorString(cuErrSync));

    cudaError_t cuErrAsync = cudaDeviceSynchronize();
    if (cuErrAsync != cudaSuccess) printf("asyc error:  %s\n", cudaGetErrorString(asyncErr0));

    checkElementsAre(7, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}