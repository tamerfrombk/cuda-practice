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
  int dim = 32;
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

  constexpr auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  dim3 threadsPerBlock(blockSize, blockSize);
  // NOTE: in this particular case, since dim == blockSize, we can replace the
  // below with dim3 blockConfig(1, 1) and remove the bounds checking above. To
  // keep general though, we can use the below.
  dim3 blockConfig(ceildiv(n, threadsPerBlock.x),
                   ceildiv(n, threadsPerBlock.y));
  simple_matrix_multiply<<<blockConfig, threadsPerBlock>>>(d_a, d_b, d_c, dim);

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
