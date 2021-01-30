// DOT PRODUCT
// https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf

extern "C"
__global__ void dot_product_float(float* a, float* b, float* c, int N, int P) {

    __shared__ float products[N];
    int id = threadIdx.x
    products[id] = a[id] * b[id];

    __syncthreads();

    if(threadIdx.x == 0) {
        float partialsum = 0;
        for(int i=0; i<N; i++) {
            partialsum += products[i];
            // Store partial sum into return array with size int(N/P)
            // Total dot product sum is calculated by host
            if ((i+1)%P == 0) {
                c[i/P] = partialsum;
                partialsum = 0;
            }
        }
    }
}

extern "C"
__global__ void dot_product_float2(float2* a, float2* b, float2* c, int N, int P) {

    __shared__ float2 products[N];
    int id = threadIdx.x
    products[id] = a[id] * b[id];

    __syncthreads();

    if(threadIdx.x == 0) {
        float2 partialsum = 0;
        for(int i=0; i<N; i++) {
            partialsum += products[i];
            // Store partial sum into return array with size N/P
            // Total dot product sum is calculated by host
            if ((i+1)%P == 0) {
                c[i/P] = partialsum;
                partialsum = 0;
            }
        }
    }
}

extern "C"
__global__ void dot_product_float4(float4* a, float4* b, float4* c, int N, int P) {

    __shared__ float4 products[N];
    int id = threadIdx.x
    products[id] = a[id] * b[id];

    __syncthreads();

    if(threadIdx.x == 0) {
        float4 partialsum = 0;
        for(int i=0; i<N; i++) {
            partialsum += products[i];
            // Store partial sum into return array with size N/P
            // Total dot product sum is calculated by host
            if ((i+1)%P == 0) {
                c[i/P] = partialsum;
                partialsum = 0;
            }
        }
    }
}