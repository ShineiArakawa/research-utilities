#pragma once

#include <cuda_runtime.h>

namespace cmath {

template <typename scalar_t>
struct vec4 {
  scalar_t x, y, z, w;

  __host__ __device__ __forceinline__ vec4() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ __forceinline__ vec4(scalar_t x, scalar_t y, scalar_t z, scalar_t w) : x(x), y(y), z(z), w(w) {}

  // 加算
  __host__ __device__ __forceinline__ vec4 operator+(const vec4& v) const {
    return vec4(x + v.x, y + v.y, z + v.z, w + v.w);
  }
  // 減算
  __host__ __device__ __forceinline__ vec4 operator-(const vec4& v) const {
    return vec4(x - v.x, y - v.y, z - v.z, w - v.w);
  }
  // スカラー乗算
  __host__ __device__ __forceinline__ vec4 operator*(scalar_t s) const {
    return vec4(x * s, y * s, z * s, w * s);
  }
  // 内積
  __host__ __device__ __forceinline__ scalar_t operator*(const vec4& v) const {
    return x * v.x + y * v.y + z * v.z + w * v.w;
  }
};

template <typename scalar_t>
struct mat4 {
  scalar_t m[4][4];

  __host__ __device__ __forceinline__ mat4() {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        m[i][j] = static_cast<scalar_t>((i == j) ? 1 : 0);
      }
    }
  }

  __host__ __device__ __forceinline__ mat4(scalar_t val) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        m[i][j] = val;
      }
    }
  }

  __host__ __device__ __forceinline__ mat4(
      scalar_t m00, scalar_t m01, scalar_t m02, scalar_t m03,
      scalar_t m10, scalar_t m11, scalar_t m12, scalar_t m13,
      scalar_t m20, scalar_t m21, scalar_t m22, scalar_t m23,
      scalar_t m30, scalar_t m31, scalar_t m32, scalar_t m33) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[0][3] = m03;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[1][3] = m13;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    m[2][3] = m23;
    m[3][0] = m30;
    m[3][1] = m31;
    m[3][2] = m32;
    m[3][3] = m33;
  }

  // 加算
  __host__ __device__ __forceinline__ mat4 operator+(const mat4& other) const {
    mat4 result;

    result.m[0][0] = m[0][0] + other.m[0][0];
    result.m[0][1] = m[0][1] + other.m[0][1];
    result.m[0][2] = m[0][2] + other.m[0][2];
    result.m[0][3] = m[0][3] + other.m[0][3];

    result.m[1][0] = m[1][0] + other.m[1][0];
    result.m[1][1] = m[1][1] + other.m[1][1];
    result.m[1][2] = m[1][2] + other.m[1][2];
    result.m[1][3] = m[1][3] + other.m[1][3];

    result.m[2][0] = m[2][0] + other.m[2][0];
    result.m[2][1] = m[2][1] + other.m[2][1];
    result.m[2][2] = m[2][2] + other.m[2][2];
    result.m[2][3] = m[2][3] + other.m[2][3];

    result.m[3][0] = m[3][0] + other.m[3][0];
    result.m[3][1] = m[3][1] + other.m[3][1];
    result.m[3][2] = m[3][2] + other.m[3][2];
    result.m[3][3] = m[3][3] + other.m[3][3];

    return result;
  }

  // 減算
  __host__ __device__ __forceinline__ mat4 operator-(const mat4& other) const {
    mat4 result;

    result.m[0][0] = m[0][0] - other.m[0][0];
    result.m[0][1] = m[0][1] - other.m[0][1];
    result.m[0][2] = m[0][2] - other.m[0][2];
    result.m[0][3] = m[0][3] - other.m[0][3];

    result.m[1][0] = m[1][0] - other.m[1][0];
    result.m[1][1] = m[1][1] - other.m[1][1];
    result.m[1][2] = m[1][2] - other.m[1][2];
    result.m[1][3] = m[1][3] - other.m[1][3];

    result.m[2][0] = m[2][0] - other.m[2][0];
    result.m[2][1] = m[2][1] - other.m[2][1];
    result.m[2][2] = m[2][2] - other.m[2][2];
    result.m[2][3] = m[2][3] - other.m[2][3];

    result.m[3][0] = m[3][0] - other.m[3][0];
    result.m[3][1] = m[3][1] - other.m[3][1];
    result.m[3][2] = m[3][2] - other.m[3][2];
    result.m[3][3] = m[3][3] - other.m[3][3];

    return result;
  }

  // スカラー乗算
  __host__ __device__ __forceinline__ mat4 operator*(scalar_t s) const {
    mat4 result;

    result.m[0][0] = m[0][0] * s;
    result.m[0][1] = m[0][1] * s;
    result.m[0][2] = m[0][2] * s;
    result.m[0][3] = m[0][3] * s;

    result.m[1][0] = m[1][0] * s;
    result.m[1][1] = m[1][1] * s;
    result.m[1][2] = m[1][2] * s;
    result.m[1][3] = m[1][3] * s;

    result.m[2][0] = m[2][0] * s;
    result.m[2][1] = m[2][1] * s;
    result.m[2][2] = m[2][2] * s;
    result.m[2][3] = m[2][3] * s;

    result.m[3][0] = m[3][0] * s;
    result.m[3][1] = m[3][1] * s;
    result.m[3][2] = m[3][2] * s;
    result.m[3][3] = m[3][3] * s;

    return result;
  }

  // 行列-ベクトル積
  __host__ __device__ __forceinline__ vec4<scalar_t> operator*(const vec4<scalar_t>& v) const {
    return vec4<scalar_t>(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w);
  }

  // 行列-行列積
  __host__ __device__ __forceinline__ mat4 operator*(const mat4& other) const {
    mat4 result;

    result.m[0][0] = m[0][0] * other.m[0][0] + m[0][1] * other.m[1][0] + m[0][2] * other.m[2][0] + m[0][3] * other.m[3][0];
    result.m[0][1] = m[0][0] * other.m[0][1] + m[0][1] * other.m[1][1] + m[0][2] * other.m[2][1] + m[0][3] * other.m[3][1];
    result.m[0][2] = m[0][0] * other.m[0][2] + m[0][1] * other.m[1][2] + m[0][2] * other.m[2][2] + m[0][3] * other.m[3][2];
    result.m[0][3] = m[0][0] * other.m[0][3] + m[0][1] * other.m[1][3] + m[0][2] * other.m[2][3] + m[0][3] * other.m[3][3];

    result.m[1][0] = m[1][0] * other.m[0][0] + m[1][1] * other.m[1][0] + m[1][2] * other.m[2][0] + m[1][3] * other.m[3][0];
    result.m[1][1] = m[1][0] * other.m[0][1] + m[1][1] * other.m[1][1] + m[1][2] * other.m[2][1] + m[1][3] * other.m[3][1];
    result.m[1][2] = m[1][0] * other.m[0][2] + m[1][1] * other.m[1][2] + m[1][2] * other.m[2][2] + m[1][3] * other.m[3][2];
    result.m[1][3] = m[1][0] * other.m[0][3] + m[1][1] * other.m[1][3] + m[1][2] * other.m[2][3] + m[1][3] * other.m[3][3];

    result.m[2][0] = m[2][0] * other.m[0][0] + m[2][1] * other.m[1][0] + m[2][2] * other.m[2][0] + m[2][3] * other.m[3][0];
    result.m[2][1] = m[2][0] * other.m[0][1] + m[2][1] * other.m[1][1] + m[2][2] * other.m[2][1] + m[2][3] * other.m[3][1];
    result.m[2][2] = m[2][0] * other.m[0][2] + m[2][1] * other.m[1][2] + m[2][2] * other.m[2][2] + m[2][3] * other.m[3][2];
    result.m[2][3] = m[2][0] * other.m[0][3] + m[2][1] * other.m[1][3] + m[2][2] * other.m[2][3] + m[2][3] * other.m[3][3];

    result.m[3][0] = m[3][0] * other.m[0][0] + m[3][1] * other.m[1][0] + m[3][2] * other.m[2][0] + m[3][3] * other.m[3][0];
    result.m[3][1] = m[3][0] * other.m[0][1] + m[3][1] * other.m[1][1] + m[3][2] * other.m[2][1] + m[3][3] * other.m[3][1];
    result.m[3][2] = m[3][0] * other.m[0][2] + m[3][1] * other.m[1][2] + m[3][2] * other.m[2][2] + m[3][3] * other.m[3][2];
    result.m[3][3] = m[3][0] * other.m[0][3] + m[3][1] * other.m[1][3] + m[3][2] * other.m[2][3] + m[3][3] * other.m[3][3];

    return result;
  }
};

}  // namespace cmath