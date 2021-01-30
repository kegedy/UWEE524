#include "helper.h"

void blas2_test() {

    CUmodule cuModule;
    CUfunction cuFunction;
    char* ModuleFile2 = "blas2.ptx";
    char* KernelName2_0 = "blas2";

    const int N = 1024;
    const int THREADS_PER_BLOCK = 1024;
    const int NUMBER_OF_BLOCKS = 32;
    const int GRID_DIM_X = NUMBER_OF_BLOCKS;
    const int GRID_DIM_Y = 1;
    const int GRID_DIM_Z = 1;
    const int BLOCK_DIM_X = THREADS_PER_BLOCK;
    const int BLOCK_DIM_Y = 1;
    const int BLOCK_DIM_Z = 1;

    // Number of bytes to allocate
    int Nsize = N * sizeof(float);
    int MatSize = (N*M) * sizeof(float);

    // Load precompiled PTX from nvcc -> cuModule
    cudaChk(cuModuleLoad(&cuModule, ModuleFile));
        
    // Get function handle from module -> cuFunction
    cudaChk(cuModuleGetFunction(&cuFunction, cuModule, KernelName));

    // Allocate vectors in host memory
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);

    // Allocate vectors in device memory
    CUdeviceptr dev_a, dev_b, dev_c;
    cudaChk(cuMemAlloc(&dev_a, size));
    cudaChk(cuMemAlloc(&dev_b, size));
    cudaChk(cuMemAlloc(&dev_c, size);

    // Initialize host vectors
    initArr(3, a, N);
    initArr(4, b, N);
    initArr(0, c, N);

    // Copy vectors from host memory to device memory
    cudaChk(cuMemcpyHtoD(dev_a, a, size));
    cudaChk(cuMemcpyHtoD(dev_b, b, size));

    // setup kernel arguments (using the simple kernel argument format)
    unsigned int sharedMemBytes = 1;
    CUstream hStream = 0;
    void* args[] = { &dev_a, &dev_b, &dev_c, M, N };

    // Launch the kernel on device
    TIMER_START
    cudaChk(cuLaunchKernel(cuFunction, \
        GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, \
        BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, \
        sharedMemBytes, hStream, args, 0));
    cudaChk(cuCtxSynchronize());
    TIMER_STOP

    // Check for errors
    cudaError_t cuErrSync  = cudaGetLastError();
    if (cuErrSync != cudaSuccess) printf("sync error: %s\n", cudaGetErrorString(cuErrSync));
    cudaError_t cuErrAsync = cudaDeviceSynchronize();
    if (cuErrAsync != cudaSuccess) printf("asyc error:  %s\n", cudaGetErrorString(asyncErr0));

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
}