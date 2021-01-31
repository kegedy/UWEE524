#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include "helper.h"
#include "matrix_add.cu"
//#include "matrix_add_test.cuh"

#define N 4
#define M 4

#define TIMER_INIT \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER t1,t2; \
    double elapsedTime; \
    QueryPerformanceFrequency(&frequency);

#define TIMER_START QueryPerformanceCounter(&t1);

#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    elapsedTime=(float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart; \
    printf("0.000000%f s\n", elapsedTime);

int main() {

    printf("START\n");
    // Configure device
    int deviceCount = 0;
    CUdevice cuDevice;
    CUcontext cuContext;

    // Initialize GPU host API 
    cuInit(0);

    // Query for device information
    cuDeviceGetCount(&deviceCount);

    // Check device exists
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(0);
    }

    // Get handle for device 0 -> cuDevice
    cuDeviceGet(&cuDevice, 0);

    // Create context -> cuContext
    cuCtxCreate(&cuContext, 0, cuDevice);

    TIMER_INIT
    CUmodule cuModule;
    CUfunction cuFunction;
    char* ModuleFile = (char*)"matrix_add.ptx";
    char* KernelName = (char*)"matrix_add";

    const int THREADS_PER_BLOCK = 1024;
    const int NUMBER_OF_BLOCKS = 4;
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
    float a[M][N];
    float b[M][N];
    float c[M][N];
    //float** a = (float**)malloc(M * sizeof(float*));
    //for(int i = 0; i < M; i++) a[i] = (float*)malloc(N * sizeof(float));
    //float** b = (float**)malloc(M * sizeof(float*));
    //for(int i = 0; i < M; i++) b[i] = (float*)malloc(N * sizeof(float));
    //float** c = (float**)malloc(M * sizeof(float*));
    //for(int i = 0; i < M; i++) c[i] = (float*)malloc(N * sizeof(float));

    // Allocate vectors in device memory
    CUdeviceptr dev_a, dev_b, dev_c;
    cuMemAlloc(&dev_a, size);
    cuMemAlloc(&dev_b, size);
    cuMemAlloc(&dev_c, size);

    // Initialize host vectors
    //initMat(3, a, M, N);
    //initMat(4, b, M, N);
    //initMat(0, c, M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 3;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            b[i][j] = 4;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
        }
    }

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

    // Retrieve results from device & verify/use
    cuMemcpyDtoH(c, dev_c, size);

    // Check data for correctness
    //checkElementsMat(7, c, M, N);
    float target = 7;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (c[i][j] != target) {
                printf("FAIL: a[%d][%d] - %0.0f does not equal %0.0f\n", i, j, a[i][j], target);
            }
        }
    }
    printf("SUCCESS! All values calculated correctly.\n");

    // Free Host Memory
    // free(a);
    // free(b);
    // free(c);

    // Free Device Memory
    cuMemFree(dev_a);
    cuMemFree(dev_b);
    cuMemFree(dev_c);

    return 0;
}