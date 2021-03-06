#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix_add_test.h"
#include "dot_product_test.h"
#include "blas2_test.h"
#include "helper.h"

int main(int argc, char** argv) {
    
    // Define static variables
    int deviceCount = 0;
    CUdevice cuDevice;
    CUcontext cuContext;
    TIMER_INIT
   
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

    // Tests
    printf('matrix_add tests:');
    matrix_add_test();
    // printf('\n')
    // printf('dot_product tests:');
    // dot_product_test();
    // printf('\n');
    // printf('blas2 tests:');
    // blas2_test();

    return 0;
}