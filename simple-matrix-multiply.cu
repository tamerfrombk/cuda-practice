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

  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);

  for (int i = 0; i < n; i++) {
    hm.a[i] = i * 1.0f;
    hm.b[i] = i * 2.0f;
  }

  auto dm = ctx.upload_inputs_to_device(hm);

  int blockSize = 32;

  dim3 threadsPerBlock(blockSize, blockSize);
  dim3 blockConfig(ceildiv(dim, blockSize), ceildiv(dim, blockSize));
  RUN_KERNEL(simple_matrix_multiply, dm.a, dm.b, dm.c, dim, blockConfig,
             threadsPerBlock);

  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[n - 1]);
}
