#pragma once

#include <cuda_common.cuh>

// ---------------------------------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------------------------------

#if defined(__CUDACC__)
template <typename scalar_t>
__global__ void calc_radial_psd_profile_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,  // (batch_size, input_height, input_width, num_channels)
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> psd           // (batch_size, n_divs, n_points, num_channels)
);
#endif

// ---------------------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------------------
void launch_calc_radial_psd_profile_kernel(const at::Tensor &input,
                                           at::Tensor &psd);
