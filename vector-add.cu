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

// This is a CUDA program designed to add two vectors into an input vector.
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

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // std::ciel
  vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

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

  printf("c[0] = %f\n", h_c[0]);
  printf("c[1023] = %f\n", h_c[1023]);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
