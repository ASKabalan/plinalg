#include "nanobind/nanobind.h"
#include "hermitian_kernel.h" // Add this line
#include "kernel_helpers.h" // Add this line
#include "kernel_nanobind_helpers.h"
namespace plinalg {

namespace nb = nanobind;


std::string BuildHermitianDescriptor(std::int64_t batch_size,
                                     std::int64_t m_rows,
                                     std::int64_t n_cols) {
  return PackDescriptorAsString(HermitianDescriptor{batch_size, m_rows, n_cols});
}

nb::dict Registrations() {
  nb::dict dict;
  dict["hermitian_operator"] =
      EncapsulateFunction(Hermitian);
  return dict;
}

NB_MODULE(_pHermitian, m) {
  m.def("registrations", &Registrations);
  m.def("build_hermitian_descriptor",
        [](std::int64_t batch_size, std::int64_t m_rows, std::int64_t n_cols) {
          std::string result = BuildHermitianDescriptor(batch_size, m_rows, n_cols);
          return nb::bytes(result.data(), result.size());
        });
}

} // namespace plinalg