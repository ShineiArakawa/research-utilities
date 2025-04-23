#pragma once

#include <cuda_common.cuh>

void clip_image(
    const at::Tensor& input,  // [batch_size, input_height, input_width, num_channels]
    at::Tensor& output        // [batch_size, output_height, output_width, num_channels]
);

// ---------------------------------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------------------------------

#if defined(__CUDACC__)
template <typename scalar_t>
static __device__ __forceinline__ int4 get_neighbor_pixel_ids(
    const scalar_t u,
    const scalar_t v,
    const int input_width,
    const int input_height) {
  const int x_idx_low = floor(u * static_cast<scalar_t>(input_width) - 0.5);   // [-1, input_width - 1]
  const int y_idx_low = floor(v * static_cast<scalar_t>(input_height) - 0.5);  // [-1, input_height - 1]
  const int x_idx_high = x_idx_low + 1;                                        // [0, input_width]
  const int y_idx_high = y_idx_low + 1;                                        // [0, input_height]

  return make_int4(x_idx_low, y_idx_low, x_idx_high, y_idx_high);
}

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