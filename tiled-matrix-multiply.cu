#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// This is a CUDA program designed to multiply two input matrixes using tiling.
// Here we use block_size == TILE_SIZE below.

// Tiling addresses one of the main concerns of memory-bound kernels: reducing
// the number of redundant global memory accesses. By partitioning the input
// data into smaller tiles that fit into faster shared memory, threads can reuse
// shared memory locally than fetching from much slower global memory leading
// to:
// 1. Reduced Memory Traffic: fast memory access versus slow global access
// 2. Improved Bandwidth: tiling allows for memory coalescing where consective
// threads read consectutive memory locations.
// 3. Higher arithmetic intensity: By moving data to fast memory, more time is
// spend in compute.
// 4. Scalability: tiling allows kernels to handle matrices larger than the
// available shared memory.

// Compile time define TILE_SIZE
#define TILE_SIZE 16

__global__ void tiled_matrix_multiply(float *a, float *b, float *p, int n) {
  __shared__ float s_a[TILE_SIZE][TILE_SIZE];
  __shared__ float s_b[TILE_SIZE][TILE_SIZE];

  // Calculate the global row and column thread indices
  // Remember, blockDim.(y,x) == TILE_SIZE
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  // For each tile, load the reused portions of the matrix multiplication into
  // shared memory.
  float s = 0;
  for (int t = 0; t < n; t += TILE_SIZE) {
    // Bounds check the row is within range and that the columns do not overflow
    if (row < n and t + col < n) {
      s_a[threadIdx.y][threadIdx.x] = a[t + (row * n) + threadIdx.x];
    } else {
      // If out of bounds, use a value that provides an identity (0)
      s_a[threadIdx.y][threadIdx.x] = 0.0;
    }

    // Bounds check the column is within range and that the rows do not overflow
    if (col < n and t + row < n) {
      s_b[threadIdx.y][threadIdx.x] = b[t + (threadIdx.y * n) + col];
    } else {
      s_b[threadIdx.y][threadIdx.x] = 0;
    }

    // Need a sync threads here to ensure all threads have written to shared
    // memory before reading from it below.
    __syncthreads();

    // This is the reduction loop that performs the dot product to produce one
    // output value at the specified cell.
    for (int k = 0; k < n; ++k) {
      s += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
    }

    // Need a sync threads here to ensure all threads arrive here before
    // overwriting shared memory above in the next iteration.
    __syncthreads();
  }

  // Update the output matrix after ensuring global row and column are in range.
  if (row < n and col < n) {
    p[row * n + col] = s;
  }
}

int main() {
  int dim = 512;
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

  int blockSize = TILE_SIZE;

  constexpr auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  dim3 threadsPerBlock(blockSize, blockSize);
  dim3 blockConfig(ceildiv(n, threadsPerBlock.x),
                   ceildiv(n, threadsPerBlock.y));
  tiled_matrix_multiply<<<blockConfig, threadsPerBlock>>>(d_a, d_b, d_c, dim);

  // Launches are async and report nothing through <<<>>>, so check explicitly:
  // cudaGetLastError catches launch-time failures (bad config, no kernel image
  // for this GPU), cudaDeviceSynchronize catches faults during execution.
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Now that the kernel has run, copy back to host
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Device memory must be freed separately
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  for (int i = 0; i < n; ++i) {
    printf("%f ", h_c[i]);
  }
  putchar('\n');

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
