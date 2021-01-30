// BLAS2: y = aMv + bw where
//   M is MxN,
//   v is Nx1,
//   w is Mx1,
//   a, b are scalar floats and
//   resultvector is Mx1

extern "C"
__global__ void blas2(float* alpha, float* M, float* v, float* beta, float* w, M, N, foat* y) {

    int row  = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;

	if(row < M) {
		for(int i=0; i<N; i++) {
            sum +=  alpha*M[row*N + i] * x[i];
        }
        y[row] = sum;
        w[row] = beta* w[row];
        y[row] = y[row] + w[row];
    }
    
}