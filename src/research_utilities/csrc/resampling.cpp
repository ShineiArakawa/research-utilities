#include <torch/extension.h>

#include <common.hpp>
#include <resampling.cuh>

// ----------------------------------------------------------------------------
// PyTorch interface
// ----------------------------------------------------------------------------
enum class InterpMethod {
  NEAREST = 0,
  BILINEAR = 1,
  BICUBIC = 2,
  LANCZOS4 = 3,
};

at::Tensor resample_image_cuda(
    const at::Tensor& input,
    const at::Tensor& grid,
    const enum InterpMethod interp_method) {
  // Check input ========================================
  CHECK_INPUT(input);
  CHECK_INPUT(grid);

  // Input expected to be 4D tensor: (B, H, W, C)
  TORCH_INTERNAL_ASSERT(input.ndimension() == 4);

  // Grid expected to be 4D tensor: (B, H', W', 2)
  TORCH_INTERNAL_ASSERT(grid.ndimension() == 4);
  TORCH_INTERNAL_ASSERT(grid.size(3) == 2);
  // ====================================================

  at::Tensor output = torch::empty({input.size(0), grid.size(1), grid.size(2), input.size(3)}, input.options());

  switch (interp_method) {
    case InterpMethod::NEAREST:
      // Launch CUDA kernel
      launch_nearest_interp_kernel(input, grid, output);
      break;
    case InterpMethod::BILINEAR:
      // Launch CUDA kernel
      launch_bilinear_interp_kernel(input, grid, output);
      break;
    case InterpMethod::BICUBIC:
      // Launch CUDA kernel
      launch_bicubic_interp_kernel(input, grid, output);
      break;
    case InterpMethod::LANCZOS4:
      // Launch CUDA kernel
      launch_lanczos_4_interp_kernel(input, grid, output);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown interpolation method");
  }

  return output;
}

// ----------------------------------------------------------------------------
// Python interface
// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Generate documentation
  std::stringstream doc_stream;
  doc_stream << "A module for resampling operations (CUDA)" << std::endl;
  doc_stream << std::endl;
  doc_stream << "(Compiled on " << __DATE__ << " at " << __TIME__ << ", with ";
#if defined(__clang__)
  doc_stream << "Clang version " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
  doc_stream << "GCC version " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(_MSC_VER)
  doc_stream << "MSVC version " << _MSC_VER;
#else
  doc_stream << "Unknown compiler";
#endif
  doc_stream << ")" << std::endl;
  doc_stream << std::endl;
  doc_stream << "[Definitions]" << std::endl;
  doc_stream << "WITH_CUDA: ";
#if defined(WITH_CUDA)
  doc_stream << "YES";
#else
  doc_stream << "NO";
#endif
  doc_stream << std::endl;
  doc_stream << "WITH_OPENMP: ";
#if defined(WITH_OPENMP)
  doc_stream << "YES";
#else
  doc_stream << "NO";
#endif
  doc_stream << std::endl;

  m.doc() = doc_stream.str().c_str();

  // Interpolation methods
  py::enum_<InterpMethod>(m, "InterpMethod")
      .value("NEAREST", InterpMethod::NEAREST)
      .value("BILINEAR", InterpMethod::BILINEAR)
      .value("BICUBIC", InterpMethod::BICUBIC)
      .value("LANCZOS4", InterpMethod::LANCZOS4);

  // Methods
  m.def("resample_image_cuda", &resample_image_cuda, "Resample image using CUDA");
}
