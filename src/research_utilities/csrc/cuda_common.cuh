#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <string>

#define THREADS_X 16
#define THREADS_Y 16

// Utility function to check for errors in CUDA calls
#define CUDA_CHECK_ERRORS(ans)             \
  {                                        \
    gpu_assert((ans), __FILE__, __LINE__); \
  }

static void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << ("GPUassert: : " + std::string(cudaGetErrorString(code)) + " " + std::string(file) + " " + std::to_string(line));
    if (abort) {
      std::exit(code);
    }
  }
};

static int div_round_up(int x, int y) {
  return (x + y - 1) / y;
};