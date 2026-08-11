#include "cuda-utilities.hpp"
#include <stdio.h>

// This is a CUDA program designed to multiply two input matrixes using tiling.
// Here we use block_size == TILE_SIZE below.

// Tiling addresses one of the main concerns of memory-bound kernels: reducing
// the number of redundant global memory accesses. By partitioning the input
// data into smaller tiles that fit into faster shared memory, threads can reuse
// shared memory locally than fetching from much slower global memory leading
// to:
// 1. Reduced Memory Traffic: fast memory access versus slow global access
// 2. Improved Bandwidth: tiling allows for memory coalescing where consective
// threads read consecutive memory locations.
// 3. Higher arithmetic intensity: By moving data to fast memory, more time is
// spend in compute.
// 4. Scalability: tiling allows kernels to handle matrices larger than the
// available shared memory.

// Compile time define TILE_SIZE
#define TILE_SIZE 32

__global__ void tiled_matrix_multiply(float *a, float *b, float *p, int n) {
  __shared__ float s_a[TILE_SIZE][TILE_SIZE];
  __shared__ float s_b[TILE_SIZE][TILE_SIZE];

  // Calculate the global row and column thread indices
  // Remember, blockDim.(y,x) == TILE_SIZE
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // For each tile, load the reused portions of the matrix multiplication into
  // shared memory.
  float s = 0;
  for (int t = 0; t < n; t += TILE_SIZE) {
    int aCol = t + threadIdx.x;
    int bRow = t + threadIdx.y;
    s_a[threadIdx.y][threadIdx.x] =
        (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
    s_b[threadIdx.y][threadIdx.x] =
        (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

    // Need a sync threads here to ensure all threads have written to shared
    // memory before reading from it below.
    __syncthreads();

    // This is the reduction loop that performs the dot product to produce one
    // output value at the specified cell.
    for (int k = 0; k < TILE_SIZE; ++k) {
      s += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
    }

    // Need a sync threads here to ensure all threads arrive here before
    // overwriting shared memory above in the next iteration.
    __syncthreads();
  }

  if (row < n && col < n) {
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

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blockConfig(ceildiv(dim, threadsPerBlock.x),
                   ceildiv(dim, threadsPerBlock.y));
  RUN_KERNEL(tiled_matrix_multiply, dm.a, dm.b, dm.c, dim, blockConfig,
             threadsPerBlock);

  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[n - 1]);
}
