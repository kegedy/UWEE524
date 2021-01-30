# HW01
### Kevin Egedy

### Repo
`git clone git@github.com:kegedy/UWEE524.git`

### Compile CUDA kernels
`nvcc -o home/kegedy/Documents/UWEE524/hw/kevin/hw01/driverAPI/kernels/matrix_add.ptx home/kegedy/Documents/UWEE524/hw/kevin/hw01/driverAPI/kernels/matrix_add.cu -ptx`
`nvcc -o home/kegedy/Documents/UWEE524/hw/kevin/hw01/driverAPI/kernels/dot_product.ptx home/kegedy/Documents/UWEE524/hw/kevin/hw01/driverAPI/kernels/dot_product.cu -ptx`
`nvcc -o home/kegedy/Documents/UWEE524/hw/kevin/hw01/driverAPI/kernels/blas2.ptx home/kegedy/Documents/UWEE524/hw/kevin/hw01/driverAPI/kernels/blas2.cu -ptx`

### Questions
Q1. See comments in matrix_add.cu, lines 8-10.

Q2. See comments in host.cpp, lines 127 and 133.

Q3. cuCtxSynchronize() blocks until the device has completed all preceding requested tasks. Time must be captured after this in order to accurately measure the time of the task.

Q4. Using performance counters around the kernel dispatch is not satisfactory because the time is captured at the CPU instead of the GPU.

Q5. Better timing measurements are [PROVIDE]

Q6. product_add.cu results
| Run  | precision: float | precision: float2 | precision: float4 |
|------|------|------|------|
|  1   | 0.0  | 0.0  | 0.0  |
|  2   | 0.0  | 0.0  | 0.0  |
|  3   | 0.0  | 0.0  | 0.0  |
|  4   | 0.0  | 0.0  | 0.0  |
|  5   | 0.0  | 0.0  | 0.0  |
|------|------|------|------|
| avg  | 0.0  | 0.0  | 0.0  |

Q7. Kernel tolerances
| Tolerance | kernel: matrix_add | kernel: dot_product | kernel: blas2 |
|-----------|-------|-------|-------|
| 1e-7      | pass  | pass  | pass  |
| 1e-8      | pass  | pass  | pass  |
| 1e-9      | pass  | pass  | pass  |
| 1e-10     | pass  | pass  | pass  |
| 1e-11     | pass  | pass  | pass  |


Q8. Kernel [PROVIDE] had the greatest difference in results between CPU and GPU.