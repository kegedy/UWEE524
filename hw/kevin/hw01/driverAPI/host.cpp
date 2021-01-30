#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tests.h"

#define N (1<<25)
#define THREADS_PER_BLOCK 1024
#define NUMBER_OF_BLOCKS 32
#define GRID_DIM_X 4
#define GRID_DIM_Y 1
#define GRID_DIM_Z 1
#define BLOCK_DIM_X THREADS_PER_BLOCK
#define BLOCK_DIM_Y 1
#define BLOCK_DIM_Z 1

// TIME OPERATIONS: 
// https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
#define TIMER_INIT \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER t1,t2; \
    double elapsedTime; \
    QueryPerformanceFrequency(&frequency);

#define TIMER_START QueryPerformanceCounter(&t1);

#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    elapsedTime=(float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart; \
    std::wcout<<elapsedTime<<L" sec"<<endl;

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main(int argc, char** argv) {
    
    TIMER_INIT

    // Define static variables
    char* module_file = "vec_add.ptx";
    char* kernel_name = "vec_add";
    int size = N * sizeof(int);
    int deviceCount = 0;
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
   
    // Initialize GPU host API 
    cudaChk(cuInit(0));
    
    // Query for device information
    cudaChk(cuDeviceGetCount(&deviceCount));

    // Check device exists
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(0);
    }

    // Get handle for device 0 -> cuDevice
    cudaChk(cuDeviceGet(&cuDevice, 0));

    // Create context -> cuContext
    cudaChk(cuCtxCreate(&cuContext, 0, cuDevice));

    // Load precompiled PTX from nvcc -> cuModule
    cudaChk(cuModuleLoad(&cuModule, module_file));
    
    // Get function handle from module -> cuFunction
    cudaChk(cuModuleGetFunction(&cuFunction, cuModule, kernel_name));

    // Allocate vectors in host memory
    float *a, *b, *c;  
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);

    // Allocate vectors in device memory
    CUdeviceptr dev_a, dev_b, dev_c;
    cudaChk(cuMemAlloc(&d_a, size));
    cudaChk(cuMemAlloc(&d_b, size));
    cudaChk(cuMemAlloc(&d_c, size));

    // Initialize host vectors
    initWidth(3, a, N);
    initWidth(4, b, N);
    initWidth(0, c, N);

    // Copy vectors from host memory to device memory
    cudaChk(cuMemcpyHtoD(dev_a, a, size));
    cudaChk(cuMemcpyHtoD(dev_b, b, size));

    // setup kernel arguments (using the simple kernel argument format)
    unsigned int sharedMemBytes = 1;
    CUstream hStream = 0;
    void* args[] = { &dev_a, &dev_b, &dev_c, &N };
    
    // Launch the kernel on device
    TIMER_START
    cudaChk(cuLaunchKernel(cuFunction, \
        GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, \
        BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, \
        sharedMemBytes, hStream, args, 0));
    cudaChk(cuCtxSynchronize());
    TIMER_STOP

    // Retrieve results from device & verify/use
    cudaChk(cuMemcpyDtoH(c, dev_c, size));

    // Check data for correctness
    checkElementsAre(7, c, N);

    // Free Host Memory
    free( a ); 
    free( b ); 
    free( c );

    // Free Device Memory
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );

    return 0;
}