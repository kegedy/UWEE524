/*
int main( ) {
    // define/init static/const variables
    // initialize GPU host API
    // query for platform/device information
    // setup GPU host API environment and device program(s)
    // allocate host memory variables h_
    // initialize host memory vars
    // allocate device memory vars
    // set up kernel arguments on device
    // copy host memory to device memory
    // determine GPU device kernel execution configuration
    // launch kernel on device
    // wait for kernel execution to complete, check for errors
    // retrieve results from device
    // use/check results
}
*/

// Cuda References: 
//    http://www.icl.utk.edu/~mgates3/docs/cuda.html
//    http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf


// DOT PRODUCT
// https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf

__global__ void dot_prod_float(float* a, float* b, float* c, int N, int P) {

    __shared__ float products[N];
    int id = threadIdx.x
    products[id] = a[id] * b[id];

    // __syncthreads();

    if(threadIdx.x == 0) {
        float partialsum = 0;
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

__global__ void dot_prod_float2(float2* a, float2* b, float2* c, int N, int P) {

    __shared__ float2 products[N];
    int id = threadIdx.x
    products[id] = a[id] * b[id];

    // __syncthreads();

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

__global__ void dot_prod_float4(float4* a, float4* b, float4* c, int N, int P) {

    __shared__ float4 products[N];
    int id = threadIdx.x
    products[id] = a[id] * b[id];

    // __syncthreads();

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

// MATRIX ADD
// http://web.mit.edu/pocky/www/cudaworkshop/Matrix/MatrixMult.cu
// https://www.jics.utk.edu/files/images/csure-reu/2015/Tutorial/CUDA-Intro.pdf
__global__ void mat_add(int *a, int *b, int *c, int M, int N) {

	int col = blockDim.x * blockIdx.x + threadIdx.x; 
	int row = blockDim.y * blockIdx.y + threadIdx.y;
    // For MxN matrix, 
    // Row-Major: row*N + col
    // Col-Major: col*M + row [not implemented]
    if (row < M && column < N) {
        int tid =  row * N + col;
        c[tid] = a[tid] + b[tid];
    }
}

// y = aMv + bw where
//   M is MxN,
//   v is Nx1,
//   w is Mx1,
//   a, b are scalar floats and
//   resultvector is Mx1
__global__ void vec_add(float* a, float* b, float* c, int N) {
    int id = threadIdx.x;
    if (id<N) {
        c[id] = a[id] + b[id];
    }
}

__global__ void BLAS2(float* a, float* M, float* v, float* b, float* w, M, N, foat* y) {

    int row  = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;

	if(row < M) {
		for(int i=0; i<N; i++) {
            sum +=  a*M[row*N + i] * x[i];
            y[row*N + i] = sum;
		}
    }
    vec_add(y, b*w, y);
}


// https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
/** Use to init the clock */
#define TIMER_INIT \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER t1,t2; \
    double elapsedTime; \
    QueryPerformanceFrequency(&frequency);


/** Use to start the performance timer */
#define TIMER_START QueryPerformanceCounter(&t1);

/** Use to stop the performance timer and output the result to the standard stream. Less verbose than \c TIMER_STOP_VERBOSE */
#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    elapsedTime=(float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart; \
    std::wcout<<elapsedTime<<L" sec"<<endl;


int main() {
    
    float fTol = 1e-7;
    if( abs( cpuResult – gpuResult) <= fTol)
    correct = TRUE;

}