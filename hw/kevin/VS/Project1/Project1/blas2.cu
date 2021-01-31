#define R 1024 //rows
#define C 1024 //cols

extern "C"
__global__ void blas2(float* alpha, float* M, float* v, float* beta, float* w, foat* y) {

    int row  = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;

	if(row < R) {
		for(int i=0; i<C; i++) {
            sum +=  alpha*M[row*N + i] * v[i];
        }
        y[row] = sum;
        w[row] = beta* w[row];
        y[row] = y[row] + w[row];
    }
    
}