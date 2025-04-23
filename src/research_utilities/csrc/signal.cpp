#include <torch/extension.h>

#include <common.hpp>
#include <signal.cuh>

// ----------------------------------------------------------------------------
// PyTorch interface
// ----------------------------------------------------------------------------
at::Tensor calc_radial_psd_profile(
    const at::Tensor& input,
    const int n_divs,
    const int n_points) {
  // Check input ========================================
  CHECK_INPUT(input);

  // Input expected to be 4D tensor: (B, H, W, C)
  TORCH_INTERNAL_ASSERT(input.ndimension() == 4);

  // ====================================================

  // Create output tensor: (B, n_divs, n_points, C)
  at::Tensor psd = torch::empty({input.size(0), n_divs, n_points, input.size(3)}, input.options());

  // Launch CUDA kernel
  launch_calc_radial_psd_profile_kernel(input, psd);

  return psd;
}

// ----------------------------------------------------------------------------
// Python interface
// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "A module for signal processing operations (CUDA)";

  // Register the functions
  m.def("calc_radial_psd_profile", &calc_radial_psd_profile, "Calculate the radial profile of the power spectral density (PSD) of a 4D tensor");
}