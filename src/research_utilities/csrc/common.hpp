#pragma once

#include <cstdlib>
#include <iostream>

#define ASSERT_MSG(condition, message)                          \
  do {                                                          \
    if (!(condition)) {                                         \
      std::cerr << "Assertion failed: (" << #condition << ")\n" \
                << "Message         : " << message << "\n"      \
                << "File            : " << __FILE__ << "\n"     \
                << "Line            : " << __LINE__ << "\n"     \
                << "Function        : " << __func__ << "\n";    \
      std::abort();                                             \
    }                                                           \
  } while (false)

// ----------------------------------------------------------------------------
// C++ interface
// ----------------------------------------------------------------------------

#define CHECK_CUDA(x) ASSERT_MSG(x.get_device() > -1, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) ASSERT_MSG(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)