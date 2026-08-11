#include "cuda-utilities.hpp"

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

  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid(ceildiv(n, threadsPerBlock.x));

  RUN_KERNEL_MAIN(n, vector_add, n, blocksPerGrid, threadsPerBlock);
}
