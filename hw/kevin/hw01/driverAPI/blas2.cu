// BLAS2: y = aMv + bw where
//   M is MxN,
//   v is Nx1,
//   w is Mx1,
//   a, b are scalar floats and
//   resultvector is Mx1

extern "C"
__global__ void BLAS2(float* a, float* M, float* v, float* b, float* w, M, N, foat* y) {

    int row  = blockDim.y * blockIdx.y + threadIdx.y;

	if(row < M) {
		for(int i=0; i<N; i++) {
            y[row] +=  a*M[row*N + i] * x[i];
        }
        w[threadIdx.y] = b* w[threadIdx.y];
        y[threadIdx.y] = y[threadIdx.y] + w[threadIdx.y];
    }
    
}