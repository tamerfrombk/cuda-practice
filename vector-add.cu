#include "cuda-utilities.hpp"
#include <stdio.h>

// This is a CUDA program designed to add two vectors into an output vector.
// It maps one thread to one output element of the vector addition.

__global__ void vector_add(float *a, float *b, float *c, int n) {
  // i here is the global thread index in the grid
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check to ensure that thread is operating on active memory
  if (i < n) {
    // Memory is coalesced here since consecutive threads access consecutive
    // addresses
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 1024;

  int threadsPerBlock = 256;
  int blocksPerGrid = ceildiv(n, threadsPerBlock);

  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);

  for (int i = 0; i < n; i++) {
    hm.a[i] = i * 1.0f;
    hm.b[i] = i * 2.0f;
  }

  auto dm = ctx.upload_inputs_to_device(hm);

  RUN_KERNEL(vector_add, dm.a, dm.b, dm.c, n, blocksPerGrid, threadsPerBlock);

  ctx.download_result_to_host(dm, hm);
  printf("c[%d] = %f\n", n - 1, hm.c[n - 1]);
}
