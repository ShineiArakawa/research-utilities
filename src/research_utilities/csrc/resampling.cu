#include <cuda_math.cuh>
#include <resampling.cuh>

void clip_image(
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

template <typename scalar_t>
__device__ __forceinline__ scalar_t sinc(const scalar_t x, const scalar_t a) {
  const scalar_t x_abs = abs(x);

  scalar_t out = 0.0;

  if (x_abs <= 1.0) {
    const scalar_t x_abs_2 = x_abs * x_abs;
    const scalar_t x_abs_3 = x_abs_2 * x_abs;
    out = (a + 2.0) * x_abs_3 - (a + 3.0) * x_abs_2 + 1.0;
  } else if (x_abs <= 2.0) {
    const scalar_t x_abs_2 = x_abs * x_abs;
    const scalar_t x_abs_3 = x_abs_2 * x_abs;
    out = a * x_abs_3 - 5.0 * a * x_abs_2 + 8.0 * a * x_abs - 4.0 * a;
  }

  return out;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t ideal_sinc(const scalar_t x) {
  if (abs(x) <= 1e-8) {
    return 1.0;
  }

  return sin(M_PI * x) / (M_PI * x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t calc_lanczos_x_weight(const scalar_t x, const scalar_t n) {
  const scalar_t x_abs = abs(x);

  if (x_abs <= n) {
    return ideal_sinc(x) * ideal_sinc(x / n);
  }

  return 0.0;
}

template <typename scalar_t>
__global__ void nearest_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
) {
  const uint batch_idx = blockIdx.z;
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height

  const uint num_channels = input.size(3);

  const uint input_height = input.size(1);
  const uint input_width = input.size(2);

  const uint output_height = grid.size(1);
  const uint output_width = grid.size(2);

  if (x < output_width && y < output_height) {
    const scalar_t dx = 1.0 / static_cast<scalar_t>(input_width);   // width
    const scalar_t dy = 1.0 / static_cast<scalar_t>(input_height);  // height

    // u, v are in the range [0, 1]
    const scalar_t u = grid[batch_idx][y][x][0];  // width
    const scalar_t v = grid[batch_idx][y][x][1];  // height

    const scalar_t u_norm = u * (1.0 - dx) + 0.5 * dx;  // [1 / dx, 1 - 1 / dx]
    const scalar_t v_norm = v * (1.0 - dy) + 0.5 * dy;  // [1 / dy, 1 - 1 / dy]

    const int x_idx = static_cast<int>(u_norm * static_cast<scalar_t>(input_width - 1));   // [0, input_width - 1]
    const int y_idx = static_cast<int>(v_norm * static_cast<scalar_t>(input_height - 1));  // [0, input_height - 1]

    for (int c = 0; c < num_channels; ++c) {
      output[batch_idx][y][x][c] = input[batch_idx][y_idx][x_idx][c];
    }
  }
}

template <typename scalar_t>
__global__ void bilinear_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
) {
  const uint batch_idx = blockIdx.z;
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height

  const uint num_channels = input.size(3);

  const uint input_height = input.size(1);
  const uint input_width = input.size(2);

  const uint output_height = grid.size(1);
  const uint output_width = grid.size(2);

  if (x < output_width && y < output_height) {
    const scalar_t dx = 1.0 / static_cast<scalar_t>(input_width);   // width
    const scalar_t dy = 1.0 / static_cast<scalar_t>(input_height);  // height

    // u, v are in the range [0, 1]
    const scalar_t u = grid[batch_idx][y][x][0];  // width
    const scalar_t v = grid[batch_idx][y][x][1];  // height

    const int4 neighbor_pixel_ids = get_neighbor_pixel_ids(u, v, input_width, input_height);

#if defined(DEBUG_MODE)
    assert(x_idx_low < x_idx_high);
    assert(y_idx_low < y_idx_high);
#endif

    const scalar_t x_low = (static_cast<scalar_t>(neighbor_pixel_ids.x) + 0.5) * dx;
    const scalar_t y_low = (static_cast<scalar_t>(neighbor_pixel_ids.y) + 0.5) * dy;
    const scalar_t x_high = (static_cast<scalar_t>(neighbor_pixel_ids.z) + 0.5) * dx;
    const scalar_t y_high = (static_cast<scalar_t>(neighbor_pixel_ids.w) + 0.5) * dy;

#if defined(DEBUG_MODE)
    assert(x_low < x_high);
    assert(y_low < y_high);
#endif

    const scalar_t x_alpha = (u - x_low) / (x_high - x_low);  // [0, 1]
    const scalar_t y_alpha = (v - y_low) / (y_high - y_low);  // [0, 1]

#if defined(DEBUG_MODE)
    assert(x_alpha >= 0.0 && x_alpha <= 1.0);
    assert(y_alpha >= 0.0 && y_alpha <= 1.0);
#endif

    const scalar_t w_ul = (1.0 - x_alpha) * (1.0 - y_alpha);
    const scalar_t w_ur = x_alpha * (1.0 - y_alpha);
    const scalar_t w_ll = (1.0 - x_alpha) * y_alpha;
    const scalar_t w_lr = x_alpha * y_alpha;

#if defined(DEBUG_MODE)
    assert(w_ul >= 0.0 && w_ul <= 1.0);
    assert(w_ur >= 0.0 && w_ur <= 1.0);
    assert(w_ll >= 0.0 && w_ll <= 1.0);
    assert(w_lr >= 0.0 && w_lr <= 1.0);
#endif

    const int x_px_low = max(0, min(neighbor_pixel_ids.x, static_cast<int>(input_width) - 1));
    const int y_px_low = max(0, min(neighbor_pixel_ids.y, static_cast<int>(input_height) - 1));
    const int x_px_high = max(0, min(neighbor_pixel_ids.z, static_cast<int>(input_width) - 1));
    const int y_px_high = max(0, min(neighbor_pixel_ids.w, static_cast<int>(input_height) - 1));

    // Interpolate
    for (int c = 0; c < num_channels; ++c) {
      output[batch_idx][y][x][c] = w_ul * input[batch_idx][y_px_low][x_px_low][c] +
                                   w_ur * input[batch_idx][y_px_low][x_px_high][c] +
                                   w_ll * input[batch_idx][y_px_high][x_px_low][c] +
                                   w_lr * input[batch_idx][y_px_high][x_px_high][c];
    }
  }
}

template <typename scalar_t>
__global__ void bicubic_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
) {
  const scalar_t a = -0.5;

  const uint batch_idx = blockIdx.z;
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height

  const uint num_channels = input.size(3);

  const uint input_height = input.size(1);
  const uint input_width = input.size(2);

  const uint output_height = grid.size(1);
  const uint output_width = grid.size(2);

  if (x < output_width && y < output_height) {
    const scalar_t dx = 1.0 / static_cast<scalar_t>(input_width);   // 1.0 / W
    const scalar_t dy = 1.0 / static_cast<scalar_t>(input_height);  // 1.0 / H

    // u, v are in the range [0, 1]
    const scalar_t u = grid[batch_idx][y][x][0];  // [0, 1]
    const scalar_t v = grid[batch_idx][y][x][1];  // [0, 1]

    // 4点を取得
    const int x_idx_1 = floor(u * static_cast<scalar_t>(input_width) - 0.5);
    const int x_idx_0 = x_idx_1 - 1;
    const int x_idx_2 = x_idx_1 + 1;
    const int x_idx_3 = x_idx_1 + 2;

    const int y_idx_1 = floor(v * static_cast<scalar_t>(input_height) - 0.5);
    const int y_idx_0 = y_idx_1 - 1;
    const int y_idx_2 = y_idx_1 + 1;
    const int y_idx_3 = y_idx_1 + 2;

    // 座標値に変換
    const scalar_t x_0 = (static_cast<scalar_t>(x_idx_0) + 0.5) * dx;
    const scalar_t x_1 = (static_cast<scalar_t>(x_idx_1) + 0.5) * dx;
    const scalar_t x_2 = (static_cast<scalar_t>(x_idx_2) + 0.5) * dx;
    const scalar_t x_3 = (static_cast<scalar_t>(x_idx_3) + 0.5) * dx;

    const scalar_t y_0 = (static_cast<scalar_t>(y_idx_0) + 0.5) * dy;
    const scalar_t y_1 = (static_cast<scalar_t>(y_idx_1) + 0.5) * dy;
    const scalar_t y_2 = (static_cast<scalar_t>(y_idx_2) + 0.5) * dy;
    const scalar_t y_3 = (static_cast<scalar_t>(y_idx_3) + 0.5) * dy;

    // 距離を計算
    const scalar_t x_dist_0 = (u - x_0) / dx;
    const scalar_t x_dist_1 = (u - x_1) / dx;
    const scalar_t x_dist_2 = (x_2 - u) / dx;
    const scalar_t x_dist_3 = (x_3 - u) / dx;

    const scalar_t y_dist_0 = (v - y_0) / dy;
    const scalar_t y_dist_1 = (v - y_1) / dy;
    const scalar_t y_dist_2 = (y_2 - v) / dy;
    const scalar_t y_dist_3 = (y_3 - v) / dy;

    // 重みを計算
    const cmath::vec4<scalar_t> h_x(sinc(x_dist_0, a),
                                    sinc(x_dist_1, a),
                                    sinc(x_dist_2, a),
                                    sinc(x_dist_3, a));

    const cmath::vec4<scalar_t> h_y(sinc(y_dist_0, a),
                                    sinc(y_dist_1, a),
                                    sinc(y_dist_2, a),
                                    sinc(y_dist_3, a));

    // 領域をクリップ
    const int x_px_0 = max(0, min(x_idx_0, static_cast<int>(input_width) - 1));
    const int x_px_1 = max(0, min(x_idx_1, static_cast<int>(input_width) - 1));
    const int x_px_2 = max(0, min(x_idx_2, static_cast<int>(input_width) - 1));
    const int x_px_3 = max(0, min(x_idx_3, static_cast<int>(input_width) - 1));

    const int y_px_0 = max(0, min(y_idx_0, static_cast<int>(input_height) - 1));
    const int y_px_1 = max(0, min(y_idx_1, static_cast<int>(input_height) - 1));
    const int y_px_2 = max(0, min(y_idx_2, static_cast<int>(input_height) - 1));
    const int y_px_3 = max(0, min(y_idx_3, static_cast<int>(input_height) - 1));

    // Interpolate
    for (int c = 0; c < num_channels; ++c) {
      const cmath::mat4<scalar_t> orig_pixel(
          input[batch_idx][y_px_0][x_px_0][c], input[batch_idx][y_px_0][x_px_1][c], input[batch_idx][y_px_0][x_px_2][c], input[batch_idx][y_px_0][x_px_3][c],
          input[batch_idx][y_px_1][x_px_0][c], input[batch_idx][y_px_1][x_px_1][c], input[batch_idx][y_px_1][x_px_2][c], input[batch_idx][y_px_1][x_px_3][c],
          input[batch_idx][y_px_2][x_px_0][c], input[batch_idx][y_px_2][x_px_1][c], input[batch_idx][y_px_2][x_px_2][c], input[batch_idx][y_px_2][x_px_3][c],
          input[batch_idx][y_px_3][x_px_0][c], input[batch_idx][y_px_3][x_px_1][c], input[batch_idx][y_px_3][x_px_2][c], input[batch_idx][y_px_3][x_px_3][c]);

      const scalar_t pixel = h_y * (orig_pixel * h_x);

      output[batch_idx][y][x][c] = pixel;
    }
  }
}

template <typename scalar_t>
__global__ void lanczos_4_interp_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,   // (batch_size, output_height, output_width, 2)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output        // (batch_size, output_height, output_width, num_channels)
) {
  const uint batch_idx = blockIdx.z;
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // height

  const uint num_channels = input.size(3);

  const uint input_height = input.size(1);
  const uint input_width = input.size(2);

  const uint output_height = grid.size(1);
  const uint output_width = grid.size(2);

  if (x < output_width && y < output_height) {
    const scalar_t dx = 1.0 / static_cast<scalar_t>(input_width);   // 1.0 / W
    const scalar_t dy = 1.0 / static_cast<scalar_t>(input_height);  // 1.0 / H

    // u, v are in the range [0, 1]
    const scalar_t u = grid[batch_idx][y][x][0];  // [0, 1]
    const scalar_t v = grid[batch_idx][y][x][1];  // [0, 1]

    const int x_ids_base = floor(u * static_cast<scalar_t>(input_width) - 0.5);   // [-1.0, input_width - 1.0]
    const int y_ids_base = floor(v * static_cast<scalar_t>(input_height) - 0.5);  // [-1.0, input_height - 1.0]

    const int x_ids[8] = {
        x_ids_base - 3,
        x_ids_base - 2,
        x_ids_base - 1,
        x_ids_base,
        x_ids_base + 1,
        x_ids_base + 2,
        x_ids_base + 3,
        x_ids_base + 4,
    };

    const int y_ids[8] = {
        y_ids_base - 3,
        y_ids_base - 2,
        y_ids_base - 1,
        y_ids_base,
        y_ids_base + 1,
        y_ids_base + 2,
        y_ids_base + 3,
        y_ids_base + 4,
    };

    const scalar_t x_coords[8] = {
        static_cast<scalar_t>(x_ids[0] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[1] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[2] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[3] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[4] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[5] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[6] + 0.5) * dx,
        static_cast<scalar_t>(x_ids[7] + 0.5) * dx,
    };

    const scalar_t y_coords[8] = {
        static_cast<scalar_t>(y_ids[0] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[1] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[2] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[3] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[4] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[5] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[6] + 0.5) * dy,
        static_cast<scalar_t>(y_ids[7] + 0.5) * dy,
    };

    const scalar_t x_dist[8] = {
        (u - x_coords[0]) / dx,
        (u - x_coords[1]) / dx,
        (u - x_coords[2]) / dx,
        (u - x_coords[3]) / dx,
        (x_coords[4] - u) / dx,
        (x_coords[5] - u) / dx,
        (x_coords[6] - u) / dx,
        (x_coords[7] - u) / dx,
    };

    const scalar_t y_dist[8] = {
        (v - y_coords[0]) / dy,
        (v - y_coords[1]) / dy,
        (v - y_coords[2]) / dy,
        (v - y_coords[3]) / dy,
        (y_coords[4] - v) / dy,
        (y_coords[5] - v) / dy,
        (y_coords[6] - v) / dy,
        (y_coords[7] - v) / dy,
    };

    // 重みを計算
    const scalar_t h_x[8] = {
        calc_lanczos_x_weight(x_dist[0], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[1], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[2], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[3], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[4], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[5], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[6], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(x_dist[7], static_cast<scalar_t>(4.0)),
    };

    const scalar_t h_y[8] = {
        calc_lanczos_x_weight(y_dist[0], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[1], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[2], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[3], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[4], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[5], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[6], static_cast<scalar_t>(4.0)),
        calc_lanczos_x_weight(y_dist[7], static_cast<scalar_t>(4.0)),
    };

    // 領域をクリップ
    const int x_px[8] = {
        max(0, min(x_ids[0], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[1], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[2], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[3], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[4], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[5], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[6], static_cast<int>(input_width) - 1)),
        max(0, min(x_ids[7], static_cast<int>(input_width) - 1)),
    };

    const int y_px[8] = {
        max(0, min(y_ids[0], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[1], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[2], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[3], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[4], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[5], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[6], static_cast<int>(input_height) - 1)),
        max(0, min(y_ids[7], static_cast<int>(input_height) - 1)),
    };

    // Interpolate
    for (int c = 0; c < num_channels; ++c) {
      // Calc h_y * (orig_pixel * h_x)

      const scalar_t tmp_pixel[8] = {
          input[batch_idx][y_px[0]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[0]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[0]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[0]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[0]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[0]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[0]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[0]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[1]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[1]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[1]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[1]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[1]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[1]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[1]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[1]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[2]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[2]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[2]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[2]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[2]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[2]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[2]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[2]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[3]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[3]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[3]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[3]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[3]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[3]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[3]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[3]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[4]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[4]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[4]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[4]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[4]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[4]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[4]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[4]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[5]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[5]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[5]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[5]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[5]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[5]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[5]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[5]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[6]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[6]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[6]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[6]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[6]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[6]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[6]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[6]][x_px[7]][c] * h_x[7],
          input[batch_idx][y_px[7]][x_px[0]][c] * h_x[0] + input[batch_idx][y_px[7]][x_px[1]][c] * h_x[1] + input[batch_idx][y_px[7]][x_px[2]][c] * h_x[2] + input[batch_idx][y_px[7]][x_px[3]][c] * h_x[3] + input[batch_idx][y_px[7]][x_px[4]][c] * h_x[4] + input[batch_idx][y_px[7]][x_px[5]][c] * h_x[5] + input[batch_idx][y_px[7]][x_px[6]][c] * h_x[6] + input[batch_idx][y_px[7]][x_px[7]][c] * h_x[7],
      };

      const scalar_t pixel = h_y[0] * tmp_pixel[0] + h_y[1] * tmp_pixel[1] + h_y[2] * tmp_pixel[2] + h_y[3] * tmp_pixel[3] + h_y[4] * tmp_pixel[4] + h_y[5] * tmp_pixel[5] + h_y[6] * tmp_pixel[6] + h_y[7] * tmp_pixel[7];

      output[batch_idx][y][x][c] = max(static_cast<scalar_t>(0.0), min(pixel, static_cast<scalar_t>(1.0)));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------------------
void launch_nearest_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output) {
  // Input shape: (batch_size, input_height, input_width, num_channels)
  const int batch_size = input.size(0);

  // Grid shape: (batch_size, output_height, output_width, 2)
  const int output_height = grid.size(1);
  const int output_width = grid.size(2);

  const dim3 threads_per_block(THREADS_X, THREADS_Y);
  const dim3 blocks_per_grid(div_round_up(output_width, threads_per_block.x),
                             div_round_up(output_height, threads_per_block.y),
                             batch_size);  // 3D grid of blocks

  // Launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "nearest_interp_kernel",
      ([&] {
        nearest_interp_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));

  // Check errors
  CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}

void launch_bilinear_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output) {
  // Input shape: (batch_size, input_height, input_width, num_channels)
  const int batch_size = input.size(0);

  // Grid shape: (batch_size, output_height, output_width, 2)
  const int output_height = grid.size(1);
  const int output_width = grid.size(2);

  const dim3 threads_per_block(THREADS_X, THREADS_Y);
  const dim3 blocks_per_grid(div_round_up(output_width, threads_per_block.x),
                             div_round_up(output_height, threads_per_block.y),
                             batch_size);  // 3D grid of blocks

  // Launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "bilinear_interp_kernel",
      ([&] {
        bilinear_interp_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));

  // Check errors
  CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}

void launch_bicubic_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output) {
  // Input shape: (batch_size, input_height, input_width, num_channels)
  const int batch_size = input.size(0);

  // Grid shape: (batch_size, output_height, output_width, 2)
  const int output_height = grid.size(1);
  const int output_width = grid.size(2);

  const dim3 threads_per_block(THREADS_X, THREADS_Y);
  const dim3 blocks_per_grid(div_round_up(output_width, threads_per_block.x),
                             div_round_up(output_height, threads_per_block.y),
                             batch_size);  // 3D grid of blocks

  // Launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "bicubic_interp_kernel",
      ([&] {
        bicubic_interp_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));

  // Check errors
  CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

  // Clip
  clip_image(input, output);
}

void launch_lanczos_4_interp_kernel(
    const at::Tensor& input,
    const at::Tensor& grid,
    at::Tensor& output) {
  // Input shape: (batch_size, input_height, input_width, num_channels)
  const int batch_size = input.size(0);

  // Grid shape: (batch_size, output_height, output_width, 2)
  const int output_height = grid.size(1);
  const int output_width = grid.size(2);

  const dim3 threads_per_block(THREADS_X, THREADS_Y);
  const dim3 blocks_per_grid(div_round_up(output_width, threads_per_block.x),
                             div_round_up(output_height, threads_per_block.y),
                             batch_size);  // 3D grid of blocks

  // Launch the kernel
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "lanczos_4_interp_kernel",
      ([&] {
        lanczos_4_interp_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));

  // Check errors
  CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

  // Clip
  clip_image(input, output);
}
