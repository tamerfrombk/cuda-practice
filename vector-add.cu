#include "cuda-utilities.hpp"

// This is a CUDA kernel designed to add two vectors into an output vector.
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

inline void run_vector_add(int n, dim3 threadsPerBlock, dim3 blocksPerGrid) {
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

// This is a CUDA kernel designed to add two vectors into an output vector.
// It maps one thread to 4 output elements (sizeof(float4) / sizeof(float))
// using the float4 CUDA type. This improves memory coalescing
// as float4 vectorizes memory access allowing the GPU to load or store 4 floats
// at once compared to one in the simple vector-add example. Moreover, we can
// launch 25% of the threads that we do in the simple vector-add example.
__global__ void vector_add4(float *a, float *b, float *c, int n) {
  // i here is the global thread index in the grid scaled by number of float4
  // elements
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  // Bounds check to ensure that thread is operating on active memory
  if (i < n) {
    // Operate on 4 floats instead of 1
    float4 *a4 = (float4 *)(a + i);
    float4 *b4 = (float4 *)(b + i);
    float4 *c4 = (float4 *)(c + i);

    c4->x = a4->x + b4->x;
    c4->y = a4->y + b4->y;
    c4->z = a4->z + b4->z;
    c4->w = a4->w + b4->w;
  }
}

inline void run_vector_add4(int n, dim3 threadsPerBlock, dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i * 1.0f;
    hm.b[i] = i * 2.0f;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(vector_add4, dm.a, dm.b, dm.c, n, blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("c[%d] = %f\n", n - 1, hm.c[n - 1]);
}

int main() {
  print_cuda_properties();
  {
    int n = 1024;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(ceildiv(n, threadsPerBlock.x));
    run_vector_add(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 1024;
    dim3 threadsPerBlock(256);

    // NOTE: we only need 1/4 of the threads
    dim3 blocksPerGrid(ceildiv(n / 4, threadsPerBlock.x));
    run_vector_add4(n, threadsPerBlock, blocksPerGrid);
  }
}
