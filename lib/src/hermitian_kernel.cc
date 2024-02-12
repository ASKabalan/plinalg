
#include <string_view>
#include "hresult.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring> // For std::memcpy
#include "hermitian_kernel.h"
#include "kernel_helpers.h"
#include <complex> // Include the necessary header file
#include "cuComplex.h"

namespace plinalg {

// call Hermitian from kernel.h and unpacl opaque to HermitianDescriptor
HRESULT Hermitian(cudaStream_t stream, void** buffers, const char* opaque,
                        std::size_t opaque_len) {
  const HermitianDescriptor &d = *UnpackDescriptor<HermitianDescriptor>(opaque, opaque_len);
  LaunchHermitianKernel(stream, buffers, d);
  checkCudaErrors(cudaGetLastError());
  return S_OK;
}
}  // namespace plinalg

