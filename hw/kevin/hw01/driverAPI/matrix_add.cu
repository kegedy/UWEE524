// MATRIX ADDITION

extern "C"
__global__ void matrix_add(float* a, float* b, float* c, float M, float N) {

	float col = blockDim.x * blockIdx.x + threadIdx.x; 
	float row = blockDim.y * blockIdx.y + threadIdx.y;
    // For MxN matrix, 
    // Row-Major: row*N + col
    // Col-Major: col*M + row [not implemented]
    if (row < M && column < N) {
        float tid =  row * N + col;
        c[tid] = a[tid] + b[tid];
    }
}

