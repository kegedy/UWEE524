#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include "helper.h"
#include "matrix_add.cu"
#include "matrix_add_test.cuh"

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

int main(int argc, char** argv) {

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

    // Tests
    printf("matrix_add tests:");
    matrix_add_test();
    // printf('\n')
    // printf('dot_product tests:');
    // dot_product_test();
    // printf('\n');
    // printf('blas2 tests:');
    // blas2_test();

    return 0;
}