
#include "cuComplex.h"
#include "hermitian_kernel.h"
#include "hresult.h"
#include "kernel_helpers.h"
#include <chrono>
#include <complex> // Include the necessary header file
#include <cstring> // For std::memcpy
#include <cuda_runtime.h>
#include <iostream>
#include <string_view>
#include <vector>

using namespace plinalg;

int main(int argc, char *argv[]) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Example descriptor
  HermitianDescriptor descriptor = {10, 5,
                                    5}; // Batch size of 1, matrix size of 4

  // Pack the descriptor
  auto packedDescriptor = PackDescriptorAsString(descriptor);

  // Allocate and initialize input and output buffers
  const int matrix_size = descriptor.m_rows * descriptor.n_cols;
  const int global_size = matrix_size * descriptor.batch_size;
  std::vector<cuComplex> hostInput(global_size);

  float real_start = 0.5f * matrix_size;
  float imag_start = -0.5f * matrix_size;

  for (int j = 1; j <= descriptor.batch_size; j++)
    for (int i = 0; i < matrix_size; ++i) {
      float real = real_start - i;
      float imag = imag_start + i;
      hostInput[i * j] = make_cuComplex(real, imag);
    }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<cuComplex> hostExpected(global_size);

  for (int j = 1; j <= descriptor.batch_size; j++)
    for (int i = 0; i < descriptor.m_rows; ++i) {
      for (int k = 0; k < descriptor.n_cols; ++k) {
        cuComplex element =
            hostInput[i * descriptor.n_cols + k + (j - 1) * matrix_size];
        hostExpected[k * descriptor.m_rows + i + (j - 1) * matrix_size] =
            make_cuComplex(cuCrealf(element), -cuCimagf(element));
      }
    }

  cudaDeviceSynchronize();
  // Stop timer
  auto stop = std::chrono::high_resolution_clock::now();

  // Calculate duration
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by cpu hermitian: " << duration.count()
            << " microseconds" << std::endl;

  std::vector<cuComplex> hostOutput(global_size);

  cuComplex *deviceInput, *deviceOutput;
  cudaMalloc(&deviceInput, global_size * sizeof(cuComplex));
  cudaMalloc(&deviceOutput, global_size * sizeof(cuComplex));

  cudaMemcpy(deviceInput, hostInput.data(), global_size * sizeof(cuComplex),
             cudaMemcpyHostToDevice);

  void *buffers[2] = {deviceInput, deviceOutput};

  // Call the Hermitian function
  start = std::chrono::high_resolution_clock::now();
  HRESULT result = Hermitian(stream, buffers, packedDescriptor.c_str(),
                             packedDescriptor.size());
  if (result != S_OK) {
    std::cerr << "Hermitian kernel launch failed." << std::endl;
    return -1;
  }
  cudaDeviceSynchronize();
  // Stop timer
  stop = std::chrono::high_resolution_clock::now();

  // Calculate duration
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by cuda kernel: " << duration.count()
            << " microseconds" << std::endl;
  // Retrieve and print the result
  cudaMemcpy(hostOutput.data(), deviceOutput, global_size * sizeof(cuComplex),
             cudaMemcpyDeviceToHost);

  if (false) {
      for (int batch = 0; batch < descriptor.batch_size; ++batch) {
        std::cout << "**************************" << std::endl;
        std::cout << "Starting Batch[" << batch << "]" << std::endl;
        std::cout << "**************************" << std::endl;
        for (int row = 0; row < descriptor.m_rows; ++row) {
          for (int col = 0; col < descriptor.n_cols; ++col) {
            int index = batch * descriptor.m_rows * descriptor.n_cols +
                        row * descriptor.n_cols + col;
            std::cout << "Batch[" << batch << "] Row[" << row << "] Col[" << col
                      << "] = (Re = " << cuCrealf(hostOutput[index])
                      << ", Img = " << cuCimagf(hostOutput[index]) << "i), "
                      << "Expected = (Re = " << cuCrealf(hostExpected[index])
                      << ", Img = " << cuCimagf(hostExpected[index]) << "i), "
                      << "Original Input = (Re = " << cuCrealf(hostInput[index])
                      << ", Img = " << cuCimagf(hostInput[index]) << "i)"
                      << std::endl;
          }
        }
      }
  }

  // Cleanup
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaStreamDestroy(stream);

  return 0;
}