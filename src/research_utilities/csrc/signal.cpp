#include <torch/extension.h>

#include <cmath>
#include <common.hpp>

#if defined(WITH_CUDA)
#include <signal.cuh>
#endif

#if defined(WITH_OPENMP)
#include <omp.h>
#endif

#define M_PI 3.14159265358979323846

namespace idx = torch::indexing;

template <typename scalar_t>
void calc_radial_psd_profile_cpu(
    const at::Tensor& input,  // [B, H, W, C]
    at::Tensor& psd) {
  const int input_height = input.size(1);
  const int input_width = input.size(2);

  const int n_divs = psd.size(1);
  const int n_points = psd.size(2);

  const scalar_t delta_theta = 2.0 * M_PI / n_divs;
  const scalar_t delta_r = 0.5 / (n_points - 1);

  const scalar_t dx = 1.0 / static_cast<scalar_t>(input_width);
  const scalar_t dy = 1.0 / static_cast<scalar_t>(input_height);

#if defined(WITH_OPENMP)
#pragma omp parallel for collapse(2)
  for (int i_div = 0; i_div < n_divs; ++i_div) {
    for (int i_point = 0; i_point < n_points; ++i_point) {
      const scalar_t theta = i_div * delta_theta;

      const scalar_t cos_theta = std::cos(theta);
      const scalar_t sin_theta = std::sin(theta);
#else
  for (int i_div = 0; i_div < n_divs; ++i_div) {
    const scalar_t theta = i_div * delta_theta;

    const scalar_t cos_theta = std::cos(theta);
    const scalar_t sin_theta = std::sin(theta);

    for (int i_point = 0; i_point < n_points; ++i_point) {
#endif
      const scalar_t r = i_point * delta_r;

      const scalar_t x = r * cos_theta + 0.5;  // [0, 1]
      const scalar_t y = r * sin_theta + 0.5;  // [0, 1]

      // get the neighboring pixels
      const int x_low_idx = std::floor(x * static_cast<scalar_t>(input_width) - 0.5);   // [-1, input_width - 1]
      const int y_low_idx = std::floor(y * static_cast<scalar_t>(input_height) - 0.5);  // [-1, input_height - 1]
      const int x_high_idx = x_low_idx + 1;                                             // [0, input_width]
      const int y_high_idx = y_low_idx + 1;                                             // [0, input_height]

      const scalar_t x_low = (static_cast<scalar_t>(x_low_idx) + 0.5) * dx;
      const scalar_t y_low = (static_cast<scalar_t>(y_low_idx) + 0.5) * dy;
      const scalar_t x_high = (static_cast<scalar_t>(x_high_idx) + 0.5) * dx;
      const scalar_t y_high = (static_cast<scalar_t>(y_high_idx) + 0.5) * dy;

      // get the weights
      const scalar_t x_alpha = (x - x_low) / (x_high - x_low);
      const scalar_t y_alpha = (y - y_low) / (y_high - y_low);

      const scalar_t w_ul = (1.0 - x_alpha) * (1.0 - y_alpha);
      const scalar_t w_ur = x_alpha * (1.0 - y_alpha);
      const scalar_t w_ll = (1.0 - x_alpha) * y_alpha;
      const scalar_t w_lr = x_alpha * y_alpha;

      // clamp the indices
      const int x_px_low = std::max(0, std::min(x_low_idx, input_width - 1));
      const int y_px_low = std::max(0, std::min(y_low_idx, input_height - 1));
      const int x_px_high = std::max(0, std::min(x_high_idx, input_width - 1));
      const int y_px_high = std::max(0, std::min(y_high_idx, input_height - 1));

      // interpolate the values
      const at::Tensor pixel_value = w_ul * input.index({idx::Slice(), y_px_low, x_px_low, idx::Slice()}) +
                                     w_ur * input.index({idx::Slice(), y_px_low, x_px_high, idx::Slice()}) +
                                     w_ll * input.index({idx::Slice(), y_px_high, x_px_low, idx::Slice()}) +
                                     w_lr * input.index({idx::Slice(), y_px_high, x_px_high, idx::Slice()});  // [B, C]

      // Set the pixel value to the output tensor
      psd.index_put_({idx::Slice(), i_div, i_point, idx::Slice()}, pixel_value);
    }
  }
}

// ----------------------------------------------------------------------------
// PyTorch interface
// ----------------------------------------------------------------------------
at::Tensor calc_radial_psd_profile(
    const at::Tensor& input,
    const int n_divs,
    const int n_points) {
  // Check input ========================================
  ASSERT_MSG(input.is_contiguous(), "Input tensor must be contiguous");
  ASSERT_MSG(input.dim() == 4, "Input tensor must be 4D");
  // ====================================================

  // Create output tensor: (B, n_divs, n_points, C)
  at::Tensor psd = torch::empty({input.size(0), n_divs, n_points, input.size(3)}, input.options());

#if defined(WITH_CUDA)
  // Launch CUDA kernel
  launch_calc_radial_psd_profile_kernel(input, psd);
#else
  // CPU implementation
  calc_radial_psd_profile_cpu<float>(input, psd);
#endif

  return psd;
}

// ----------------------------------------------------------------------------
// Python interface
// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Generate documentation
  std::stringstream doc_stream;
  doc_stream << "A module for signal processing operations" << std::endl;
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

  // Register the functions
  m.def("calc_radial_psd_profile", &calc_radial_psd_profile, "Calculate the radial profile of the power spectral density (PSD) of a 4D tensor");
}