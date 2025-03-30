#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#define THREADS_X 16
#define THREADS_Y 16

// Utility function to check for errors in CUDA calls
#define CUDA_CHECK_ERRORS(ans)             \
  {                                        \
    gpu_assert((ans), __FILE__, __LINE__); \
  }

inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << ("GPUassert: : " + std::string(cudaGetErrorString(code)) + " " + std::string(file) + " " + std::to_string(line));
    if (abort) {
      exit(code);
    }
  }
};

inline int div_round_up(int x, int y) {
  return (x + y - 1) / y;
};

inline void clip_image(
    const at::Tensor& input,  // [batch_size, input_height, input_width, num_channels]
    at::Tensor& output        // [batch_size, output_height, output_width, num_channels]
) {
  // Clip
  const auto [min, argmin] = torch::min(input.flatten(1, -1), -1, true);            // [batch_size, 1]
  const auto [max, argmax] = torch::max(input.flatten(1, -1), -1, true);            // [batch_size, 1]
  output.clamp_(min.unsqueeze(-1).unsqueeze(-1), max.unsqueeze(-1).unsqueeze(-1));  // [batch_size, output_height, output_width, num_channels]
};

// ---------------------------------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------------------------------

#if defined(__CUDACC__)
template <typename scalar_t>
__global__ void nearest_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
);

template <typename scalar_t>
__global__ void bilinear_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
);

template <typename scalar_t>
__global__ void bicubic_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
);

template <typename scalar_t>
__global__ void lanczos_4_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
);
#endif

// ---------------------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------------------
void launch_nearest_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output);

void launch_bilinear_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output);

void launch_bicubic_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output);

void launch_lanczos_4_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output);