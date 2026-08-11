#include "cuda-utilities.hpp"
#include <stdio.h>

// This is a CUDA program designed to multiply two input matrixes using the
// simplest approach:
// 1. square matrix
// 2. one thread per matrix output element
// 3. Simple bounds checking to ensure the global row and column are within
// memory.
//
// As the name says, this is the simplest approach but is a memory-bound
// operation since we re-read elements of (b) many times. This is fixed by using
// shared memory and tiling to reduce global memory bandwidth usage.

__global__ void simple_matrix_multiply(float *a, float *b, float *p, int n) {
  // Calculate the global row and column thread indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n and col < n) {
    // This is the reduction loop that performs the dot product to produce one
    // output value at the specified cell.
    float s = 0;
    for (int k = 0; k < n; ++k) {
      s += a[row * n + k] * b[k * n + col];
    }

    // Update the output matrix
    p[row * n + col] = s;
  }
}

int main() {
  int dim = 256;
  int n = dim * dim;
  size_t bytes = n * sizeof(float);

  // Use h_prefix to denote host memory
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);

  for (int i = 0; i < n; i++) {
    h_a[i] = i * 1.0f;
    h_b[i] = i * 2.0f;
  }

  // d_ prefix for device
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // Copy from host to device since CUDA operates on GPU memory
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  int blockSize = 32;

  dim3 threadsPerBlock(blockSize, blockSize);
  dim3 blockConfig(ceildiv(dim, blockSize), ceildiv(dim, blockSize));
  RUN_KERNEL(simple_matrix_multiply, d_a, d_b, d_c, dim, blockConfig,
             threadsPerBlock);

  // Now that the kernel has run, copy back to host
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Device memory must be freed separately
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  printf("%f\n", h_c[n - 1]);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
