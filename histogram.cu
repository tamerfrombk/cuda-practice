#include <cassert>
#include <cstdint>

#include "cuda-utilities.hpp"

// There are 256 possible pixel values in a grayscale image.
static constexpr int GRAYSCALE_BIN_COUNT = 256;

// This is a very simple histogram CUDA kernel. It takes a 1D array of image
// pixels assumed to be single channel grayscale (0-255) and counts the
// occurrence of each pixel value.
//
// PRECONDITIONS:
//  1. One thread is assigned to each input pixel.
//  2. `counts` is at least 256 elements large.
__global__ void simple_count_grayscale(float *img, float *, float *counts,
                                       int n) {
  // i here is the global thread index in the grid
  // One thread per input pixel
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check to ensure that thread is operating on active memory
  if (i < n) {
    const auto b = static_cast<std::uint8_t>(img[i]);
    // The atomicAdd is a CUDA builtin that atomically increases the value in
    // the provided pointer.
    // We need to atomically increment the count because many input threads
    // _could_ process the same input value (0-255) which would lead to several
    // threads trying to update the same memory address.
    atomicAdd(&counts[b], 1.0f);
  }
}

inline void run_simple_count_grayscale(int n, dim3 threadsPerBlock,
                                       dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i % GRAYSCALE_BIN_COUNT;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(simple_count_grayscale, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  // We expect that each grayscale cell holds a count of 4.
  for (int i = 0; i < GRAYSCALE_BIN_COUNT; ++i) {
    printf("%d ", (int)hm.c[i]);
  }
  printf("\n");
}

// This is an upgraded version of the simple grayscale kernel that demonstrates
// thread privatization: each block creates and associates a private copy of the
// histogram to every block.
__global__ void privatized_grayscale(float *img, float *bins_pool, float *bins,
                                     int n) {
  // i here is the global thread index in the grid
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // There is one group of bins per block so this thread must access its own
  // pool at the block offset.
  float *my_bin = &bins_pool[blockIdx.x * GRAYSCALE_BIN_COUNT];

  // Bounds check to ensure that thread is operating on active memory
  if (i < n) {
    const auto b = static_cast<std::uint8_t>(img[i]);
    // Multiple input threads can still write to the same address so atomicity
    // is still required. However, the level of contention here is reduced by a
    // factor that is approximately the number of active blocks across all SMs.
    atomicAdd(&my_bin[b], 1.0f);
  }
  // A sync threads here is required to ensure all threads have finished writing
  // their values before being read below.
  __syncthreads();

  // Walk the private bins and pool each bin's values into the global bins.
  for (int b = threadIdx.x; b < GRAYSCALE_BIN_COUNT; b += blockDim.x) {
    if (my_bin[b] > 0) {
      atomicAdd(&bins[b], my_bin[b]);
    }
  }
}

inline void run_privatized_grayscale(int n, dim3 threadsPerBlock,
                                     dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);

  for (int i = 0; i < n; i++) {
    hm.a[i] = i % GRAYSCALE_BIN_COUNT;
    hm.b[i] = 0;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(privatized_grayscale, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  // We expect that each grayscale cell holds a count of 4.
  for (int i = 0; i < GRAYSCALE_BIN_COUNT; ++i) {
    printf("%d ", (int)hm.c[i]);
  }
  printf("\n");
}

// This is yet another optimization of privatized_grayscale.
// Since the amount of total bins is small, we can have each block
// have its own histogram in the form of shared memory. Each block
// will update its shared memory with the counts of pixels it sees.
// This is an improvement over the previous version since we do not
// have to access relatively slow global memory to update the counts
// so memory latency is improved via shared memory.
__global__ void privatized_shared_grayscale(float *img, float *, float *bins,
                                            int n) {

  __shared__ int bins_s[GRAYSCALE_BIN_COUNT];
  for (int i = 0; i < GRAYSCALE_BIN_COUNT; ++i) {
    // Initialize all bins to 0
    bins_s[i] = 0;
  }
  // Sync threads required here to ensure all threads have written their values
  // before using them below.
  __syncthreads();

  // i here is the global thread index in the grid
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check to ensure that thread is operating on active memory
  if (i < n) {
    const auto b = static_cast<std::uint8_t>(img[i]);
    // Multiple input threads can still write to the same address so atomicity
    // is still required.
    atomicAdd(&bins_s[b], 1.0f);
  }
  // A sync threads here is required to ensure all threads have finished writing
  // their values to shared memory before being read below.
  __syncthreads();

  // Walk the private bins and pool each bin's values into the global bins.
  for (int b = threadIdx.x; b < GRAYSCALE_BIN_COUNT; b += blockDim.x) {
    if (bins_s[b] > 0) {
      atomicAdd(&bins[b], bins_s[b]);
    }
  }
}

inline void run_privatized_shared_grayscale(int n, dim3 threadsPerBlock,
                                            dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);

  for (int i = 0; i < n; i++) {
    hm.a[i] = i % GRAYSCALE_BIN_COUNT;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(privatized_shared_grayscale, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  // We expect that each grayscale cell holds a count of 4.
  for (int i = 0; i < GRAYSCALE_BIN_COUNT; ++i) {
    printf("%d ", (int)hm.c[i]);
  }
  printf("\n");
}

// This is yet another optimization of privatized_shared_grayscale
// demonstrating thread coarsening. Here, we launch the kernel with
// 32 threads per block and a single block so each thread will process
// multiple elements at once. In this case, it's 32 elements per thread.
__global__ void privatized_shared_coarsened_grayscale(float *img, float *,
                                                      float *bins, int n) {

  __shared__ int bins_s[GRAYSCALE_BIN_COUNT];
  for (int b = 0; b < GRAYSCALE_BIN_COUNT; ++b) {
    // Initialize all bins to 0
    bins_s[b] = 0;
  }
  // Sync threads required here to ensure all threads have written their values
  // before using them below.
  __syncthreads();

  int stride = blockDim.x * gridDim.x;

  // i here is the global thread index in the grid
  int i = (blockIdx.x * blockDim.x + threadIdx.x);

  // Bounds check to ensure that thread is operating on active memory
  if (i < n) {
    for (int x = i; x < n; x += stride) {
      const auto b = static_cast<std::uint8_t>(img[x]);
      // Multiple input threads can still write to the same address so atomicity
      // is still required.
      atomicAdd(&bins_s[b], 1.0f);
    }
  }
  // A sync threads here is required to ensure all threads have finished writing
  // their values to shared memory before being read below.
  __syncthreads();

  // Walk the private bins and pool each bin's values into the global bins.
  for (int b = threadIdx.x; b < GRAYSCALE_BIN_COUNT; b += blockDim.x) {
    if (bins_s[b] > 0) {
      atomicAdd(&bins[b], bins_s[b]);
    }
  }
}

inline void run_privatized_shared_coarsened_grayscale(int n,
                                                      dim3 threadsPerBlock,
                                                      dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);

  for (int i = 0; i < n; i++) {
    hm.a[i] = i % GRAYSCALE_BIN_COUNT;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(privatized_shared_coarsened_grayscale, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  // We expect that each grayscale cell holds a count of 4.
  for (int i = 0; i < GRAYSCALE_BIN_COUNT; ++i) {
    printf("%d ", (int)hm.c[i]);
  }
  printf("\n");
}

int main() {
  print_cuda_properties();
  {
    int n = 1024;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(ceildiv(n, threadsPerBlock.x));
    run_simple_count_grayscale(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 1024;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(ceildiv(n, threadsPerBlock.x));
    run_privatized_grayscale(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 1024;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(4);
    run_privatized_shared_grayscale(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 1024;
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid(1);
    run_privatized_shared_coarsened_grayscale(n, threadsPerBlock,
                                              blocksPerGrid);
  }
}
