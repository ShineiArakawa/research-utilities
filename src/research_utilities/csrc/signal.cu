#include <resampling.cuh>
#include <signal.cuh>

// ---------------------------------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------------------------------

template <typename scalar_t>
__global__ void calc_radial_psd_profile_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> psd           // (batch_size, n_divs, n_points, num_channels)
) {
  const uint batch_idx = blockIdx.z;
  const uint i_div = blockIdx.x * blockDim.x + threadIdx.x;
  const uint i_point = blockIdx.y * blockDim.y + threadIdx.y;

  const uint n_divs = psd.size(1);
  const uint n_points = psd.size(2);

  const uint input_height = input.size(1);
  const uint input_width = input.size(2);
  const uint num_channels = input.size(3);

  if (i_div < n_divs && i_point < n_points) {
    // Calculate center position
    const scalar_t delta_theta = 2.0 * M_PI / n_divs;
    const scalar_t theta = i_div * delta_theta;

    const scalar_t cos_theta = cos(theta);
    const scalar_t sin_theta = sin(theta);

    const scalar_t delta_r = 0.5 / (n_points - 1);
    const scalar_t r = i_point * delta_r;

    const scalar_t x = r * cos_theta + 0.5;  // Ranged in [0, 1]
    const scalar_t y = r * sin_theta + 0.5;  // Ranged in [0, 1]

    // Calculate the corresponding pixel value with bilinear interpolation
    const scalar_t dx = 1.0 / static_cast<scalar_t>(input_width);   // width
    const scalar_t dy = 1.0 / static_cast<scalar_t>(input_height);  // height

    const int4 neighbor_pixel_ids = get_neighbor_pixel_ids(x, y, input_width, input_height);

    const scalar_t x_low = (static_cast<scalar_t>(neighbor_pixel_ids.x) + 0.5) * dx;
    const scalar_t y_low = (static_cast<scalar_t>(neighbor_pixel_ids.y) + 0.5) * dy;
    const scalar_t x_high = (static_cast<scalar_t>(neighbor_pixel_ids.z) + 0.5) * dx;
    const scalar_t y_high = (static_cast<scalar_t>(neighbor_pixel_ids.w) + 0.5) * dy;

    const scalar_t x_alpha = (x - x_low) / (x_high - x_low);  // [0, 1]
    const scalar_t y_alpha = (y - y_low) / (y_high - y_low);  // [0, 1]

    const scalar_t w_ul = (1.0 - x_alpha) * (1.0 - y_alpha);
    const scalar_t w_ur = x_alpha * (1.0 - y_alpha);
    const scalar_t w_ll = (1.0 - x_alpha) * y_alpha;
    const scalar_t w_lr = x_alpha * y_alpha;

    const int x_px_low = max(0, min(neighbor_pixel_ids.x, static_cast<int>(input_width) - 1));
    const int y_px_low = max(0, min(neighbor_pixel_ids.y, static_cast<int>(input_height) - 1));
    const int x_px_high = max(0, min(neighbor_pixel_ids.z, static_cast<int>(input_width) - 1));
    const int y_px_high = max(0, min(neighbor_pixel_ids.w, static_cast<int>(input_height) - 1));

    // Interpolate
    for (int c = 0; c < num_channels; ++c) {
      const scalar_t pixel_value = w_ul * input[batch_idx][y_px_low][x_px_low][c] +
                                   w_ur * input[batch_idx][y_px_low][x_px_high][c] +
                                   w_ll * input[batch_idx][y_px_high][x_px_low][c] +
                                   w_lr * input[batch_idx][y_px_high][x_px_high][c];

      psd[batch_idx][i_div][i_point][c] = pixel_value * pixel_value;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------------------

void launch_calc_radial_psd_profile_kernel(
    const at::Tensor &input,
    at::Tensor &psd) {
  // Input shape: (batch_size, input_height, input_width, num_channels)

  // psd shape: (batch_size, n_divs, n_points, num_channels)
  const int n_divs = psd.size(1);
  const int n_points = psd.size(2);

  const dim3 threads_per_block(THREADS_X, THREADS_Y);
  const dim3 blocks_per_grid(div_round_up(n_divs, threads_per_block.x),
                             div_round_up(n_points, threads_per_block.y),
                             input.size(0)  // batch_size
  );

  // Launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "calc_radial_psd_profile_kernel",
      ([&] {
        calc_radial_psd_profile_kernel<scalar_t>
            <<<blocks_per_grid, threads_per_block>>>(
                input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                psd.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));

  // Check for errors
  CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}