#include <stdio.h>
#include <stdlib.h>
#include "kernel.cuh"

/*
    // STEP 0: Always check and handle errors (IMPLIED step...)
    // STEP 1: Define the platform ( = obtain the CUDA device and context)
    // STEP 2: Create and build the Module and Function
    // STEP 3: Setup memory objects to manage the input-output host and device data
    // STEP 4: Configure the kernel for execution - set up arguments, grid/index hierarchy
    // STEP 5: Launch the kernel
    // STEP 6: Retrieve results from device & verify/use

int main( ) {
    // Define static variables
    // Initialize GPU host API
    // Query for device information
    // Setup GPU host API environment and device program(s)
    // Allocate host memory variables h_
    // Initialize host memory vars
    // Allocate device memory vars d_
    // Set up kernel arguments on device
    // Copy host memory to device memory
    // Determine GPU device kernel execution configuration
    // Launch kernel on device
    // Wait for kernel execution to complete, check for errors
    // Retrieve results from device
    // Check data for correctness
    // Free Host Memory
    // Free Device Memory
}
*/

int main(int argc, char** argv)
{ 
    // Initialize
    cudaChk(cuInit(0));
    
    // Get number of devices supporting CUDA
    int deviceCount = 0;
    cudaChk(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(0);
    }

    // Get handle for device 0
    CUdevice cuDevice;
    cudaChk(cuDeviceGet(&cuDevice, 0));

    // Create context
    CUcontext cuContext;
    cudaChk(cuCtxCreate(&cuContext, 0, cuDevice));

    // Create module from binary file
    CUmodule cuModule;

    // precompiled PTX or CUBIN from nvcc
    cudaChk(cuModuleLoad(&cuModule, "vecAdd_01.ptx"));
    
    // Get function handle from module; Note mangled name
    CUfunction vecAdd;
    cudaChk(cuModuleGetFunction(&vecAdd, cuModule, "_Z9vecAdd_01PfS_S_i"));

    // Allocate input vectors h_A etc. in host memory
    float* h_A = (float*)malloc(size);

    // Allocate vectors in device memory
    CUdeviceptr d_A;
    cudaChk(cuMemAlloc(&d_A, size));

    // Copy vectors from host memory to device memory
    cudaChk(cuMemcpyHtoD(d_A, h_A, size));

    // Invoke kernel
    int threadsPerBlock = 4;
    int blocksPerGrid = 4;

    // setup kernel arguments (using the simple kernel argument format)
    void* args[] = { &d_C, &d_A, &d_B, &N };
    
    cudaChk(cuLaunchKernel(vecAdd, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0));
    cudaChk(cuCtxSynchronize());

    cudaChk(cuMemcpyDtoH(h_C, d_C, size));

    // Do data result verification routine...

    return 0;
}