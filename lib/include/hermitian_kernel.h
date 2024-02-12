

#ifndef _HERMITIAN_KERNEL_OP_H_
#define _HERMITIAN_KERNEL_OP_H_

#include <cstddef>
#include <string>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "hresult.h"

namespace plinalg {


// Make descriptor for Hermitian kernel and add the prototype for the kernel
struct HermitianDescriptor {
  std::int64_t batch_size;
  std::int64_t m_rows;
  std::int64_t n_cols;
};

void LaunchHermitianKernel(cudaStream_t stream, void** buffers,
                           HermitianDescriptor descriptor);

HRESULT Hermitian(cudaStream_t stream, void** buffers,
               const char* opaque, size_t opaque_len);;
                    
}  // namespace plinalg

#endif  // _HERMITIAN_KERNEL_OP_H_