#include "hermitian_kernel.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h> // Include for cuComplex

namespace plinalg {

__device__ void HermitianCompute(const cuComplex* input,
                                cuComplex* output,
                                const std::int64_t m_rows,
                                const std::int64_t n_cols,
                                const std::int64_t idx) {
    const std::int64_t batch = idx / (m_rows * n_cols);
    const std::int64_t row = (idx / n_cols) % m_rows;
    const std::int64_t col = idx % n_cols;
    const cuComplex value = input[idx];
    output[batch * m_rows * n_cols + col * m_rows + row] =
        make_cuComplex(cuCrealf(value), -cuCimagf(value)); // Use cuComplex functions
}

__global__ void HermitianKernel(const cuComplex* input,
                                cuComplex* output,
                                const std::int64_t batch_size,
                                const std::int64_t n_rows,
                                const std::int64_t m_cols) {
    for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < batch_size * n_rows * m_cols;
             idx += blockDim.x * gridDim.x) {
        HermitianCompute(input, output, n_rows, m_cols, idx);
    }
}

// launch the kernel using optimal grid and block size
void LaunchHermitianKernel(cudaStream_t stream, void** buffers,
                           HermitianDescriptor descriptor) {
    const cuComplex* input = reinterpret_cast<const cuComplex*>(buffers[0]);
    cuComplex* output = reinterpret_cast<cuComplex*>(buffers[1]);

    const int block_dim = 128;
    const std::int64_t grid_dim = std::min<std::int64_t>(
            1024, (descriptor.batch_size * descriptor.m_rows * descriptor.n_cols + block_dim - 1) / block_dim);

    std::cout << "CUDA : Batch size: " << descriptor.batch_size << std::endl;  
    HermitianKernel<<<grid_dim, block_dim,/*dynamic_shared_mem_bytes=*/0, stream>>>(
        input, output, descriptor.batch_size, descriptor.m_rows, descriptor.n_cols);
}

}  // namespace plinalg
