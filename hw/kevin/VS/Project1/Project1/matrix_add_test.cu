#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include "helper.h"

#define N 1024
#define M 1024

#define TIMER_INIT \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER t1,t2; \
    double elapsedTime; \
    QueryPerformanceFrequency(&frequency);

#define TIMER_START QueryPerformanceCounter(&t1);

#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    elapsedTime=(float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart; \
    printf("0.000000%f sec\n", elapsedTime);

void matrix_add_test() {

    TIMER_INIT
    CUmodule cuModule;
    CUfunction cuFunction;
    char* ModuleFile = (char*)"matrix_add.ptx";
    char* KernelName = (char*)"matrix_add";

    const int THREADS_PER_BLOCK = 1024;
    const int NUMBER_OF_BLOCKS = 32;
    const int GRID_DIM_X = 4;
    const int GRID_DIM_Y = 1;
    const int GRID_DIM_Z = 1;
    const int BLOCK_DIM_X = THREADS_PER_BLOCK;
    const int BLOCK_DIM_Y = 1;
    const int BLOCK_DIM_Z = 1;

    // Number of bytes to allocate for MxN matrix with type float
    int size = (M * N) * sizeof(float);

    // Load precompiled PTX from nvcc -> cuModule
    cuModuleLoad(&cuModule, ModuleFile);

    // Get function handle from module -> cuFunction
    cuModuleGetFunction(&cuFunction, cuModule, KernelName);

    // Allocate vectors in host memory
    float** a = (float**)malloc(size);
    float** b = (float**)malloc(size);
    float** c = (float**)malloc(size);

    // Allocate vectors in device memory
    CUdeviceptr dev_a, dev_b, dev_c;
    cuMemAlloc(&dev_a, size);
    cuMemAlloc(&dev_b, size);
    cuMemAlloc(&dev_c, size);

    // Initialize host vectors
    initMat(3, a, M, N);
    initMat(4, b, M, N);
    initMat(0, c, M, N);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(dev_a, a, size);
    cuMemcpyHtoD(dev_b, b, size);

    // setup kernel arguments (using the simple kernel argument format)
    unsigned int sharedMemBytes = 1;
    CUstream hStream = 0;
    void* args[] = { &dev_a, &dev_b, &dev_c };

    // Launch the kernel on device
    TIMER_START
        cuLaunchKernel(cuFunction, \
            GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, \
            BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, \
            sharedMemBytes, hStream, args, 0);
    cuCtxSynchronize();
    TIMER_STOP

    // Check for errors
    cudaError_t cuErrSync = cudaGetLastError();
    if (cuErrSync != cudaSuccess) printf("sync error: %s\n", cudaGetErrorString(cuErrSync));
    cudaError_t cuErrAsync = cudaDeviceSynchronize();
    if (cuErrAsync != cudaSuccess) printf("asyc error:  %s\n", cudaGetErrorString(cuErrAsync));

    // Retrieve results from device & verify/use
    cuMemcpyDtoH(c, dev_c, size);

    // Check data for correctness
    checkElementsMat(7, c, M, N);

    // Free Host Memory
    free(a);
    free(b);
    free(c);

    // Free Device Memory
    cuMemFree(dev_a);
    cuMemFree(dev_b);
    cuMemFree(dev_c);
}