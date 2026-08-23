#include "cuda-utilities.hpp"

#include <cassert>

// This is a very simple reduction kernel that sum-reduces input into its first
// element and puts the result of the reduction into output:
// The kernel has the following preconditions:
// 1. The kernel is launched with a single block
// 2. All of input (N) fits within a single block
// 3. N/2 threads are launched
// This reduction kernel works by setting a minimal stride and doubling the
// stride each loop through the reduction.
// This is not ideal for many reasons:
// 1. Control Divergence: There are two places branching on the thread index.
// This means some threads within a will execute the branch and others won't.
// Either way, the warp has to wait for both groups of threads.
// 2. Memory Coalescence: The stride starts small and consecutive threads
// _could_ read elements from the same cache line. As the stride gets large, we
// need to reach out to global memory and run into memory bandwidth issues.
// 3. __syncthreads(): We need to wait for each thread writing its contribution
// before continuing in the next iteration of the reduction.
__global__ void simple_reduction_kernel(float *input, float *, float *output,
                                        int N) {
  // N / 2 threads -- one every other output element
  int i = threadIdx.x * 2;
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    // Only threads that are a multiple of the stride contribute to the output.
    if (threadIdx.x % stride == 0) {
      // NOTE that this is a memory bound kernel: two reads, one write per 1
      // calculation per iteration.
      input[i] += input[i + stride];
    }
    // Sync all the threads here to ensure that each thread has written its
    // contribution before continuing the reduction. Otherwise, we may run into
    // a race condition where one thread could be ahead in its reduction loop
    // and overwrite another value being written into.
    __syncthreads();
  }

  // Accumulate the output in the first element of input which is assigned to
  // the first thread.
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

inline void run_simple_reduction_kernel(int n, dim3 threadsPerBlock,
                                        dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(simple_reduction_kernel, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[0]);
}

// Assigning strides to decrease significantly reduces control divergence. The
// idea is that we should arrange the threads and their own positions so that
// they can remain close to each other as the reduction loop progresses.
// With these changes, instead of adding neighbor elements in the first round,
// it adds elements that are hald a section away from each other and the section
// size is always twice the number of remaining active threads.
// There is less control divergence due to the position of the threads
// performing the addition. In the first iteration, all threads are active (all
// threads < stride) so no control divergence. In the next iteration, threads [0
// - N/2] are active; (N/2 - N] are not. The pair-wise sums are stored in the
// active threads. Since warps consist of 32 threads with consecutive
// threadIdx.x values, all threads in the first half of all warps will be active
// and all threads in the second half are inactive. Since all threads in each
// warp take the same path of execution, there is no control divergence.
// However, control divergence isn't completely eliminated. At some point,
// stride falls below 32. At that point, the final 5 iterations will have 16, 8,
// 4, 2, and 1 thread performing the addition. These iterations will have
// control divergence.
__global__ void simple_reduction_kernel_decreasing_stride(float *input, float *,
                                                          float *output,
                                                          int N) {
  // We still launch N/2 threads but note the missing *2: the owner positions of
  // all adjacent threads are now adjacent to each other.
  int i = threadIdx.x;
  for (int stride = blockDim.x; stride >= 1; stride /= 2) {
    // Only threads less than the stride contribute to the output.
    if (threadIdx.x < stride) {
      // NOTE that this is still a memory bound kernel: two reads, one write per
      // 1 calculation per iteration.
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

inline void run_simple_reduction_kernel_decreasing_stride(int n,
                                                          dim3 threadsPerBlock,
                                                          dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(simple_reduction_kernel_decreasing_stride, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[0]);
}

// This is a kernel implementing shared memory into the reduction. The first N/2
// memory reads are from global memory into shared memory. The rest of the
// reduction loops are processed in shared memory which has a much faster
// bandwidth.
__global__ void simple_shared_memory_reduction_kernel(float *input, float *,
                                                      float *output, int N) {
  const int BLOCK_DIM = 128;
  // Safety check
  assert(BLOCK_DIM == blockDim.x);

  __shared__ float input_s[BLOCK_DIM];

  int i = threadIdx.x;
  // Run the first iteration of the reduction and store the result in shared
  // memory
  input_s[i] = input[i] + input[i + blockDim.x];

  // Since the first load above used BLOCK_DIM stride, start from BLOCK_DIM / 2
  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    // Note that sync threads is here first for two reasons:
    // 1. On the first iteration of the loop, this ensures all threads have
    // written their first calculation.
    // 2. On subsequent iterations of the loop to ensure that the threads sync
    // on values values written by the previous iteration
    __syncthreads();
    // Only threads less than the stride contribute to the output.
    if (i < stride) {
      // NOTE that this is still a memory bound kernel: two reads, one write per
      // 1 calculation per iteration. However, here we are reading/writing to
      // shared memory which is much faster to access than global memory.
      input_s[i] += input_s[i + stride];
    }
  }

  if (i == 0) {
    *output = input_s[0];
  }
}

inline void run_simple_shared_memory_reduction_kernel(int n,
                                                      dim3 threadsPerBlock,
                                                      dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(simple_shared_memory_reduction_kernel, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[0]);
}

// Return the index of the warp given the thread id
// Note __device__ means we can only call this function on CUDA devices (GPUs)
__device__ int warpIdx() { return threadIdx.x / warpSize; }

// Return the position of the thread within a warp
__device__ int laneIdx() { return threadIdx.x % warpSize; }

// __shfl_down_sync is a primitive that allows a thread to send its data to
// another thread within the same warp. We use this property to implement a warp
// sum reduce: Start with the provided value, indicate all threads must be
// active in the warp (mask = 0xffffff), send the accumulated sum to the thread
// `stride` away from the current one. The net effect of this is a warp-wide
// reduction using decreasing strides.
__device__ float warp_reduce(float f) {
  float sum = f;
  // warpSize / 2 since we already completed the first iteration: we have a sum
  // to start
  for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
  }
  return sum;
}

// This is a kernel that reduces control divergence in the shared memory kernel
// through warp primitives.
// The majority of the shared memory kernel is the same execept for the loop
// boundary condition: Instead of waiting until stride falls below 32
// in the final 5 iterations of the loop, we exit the loop once the
// stride falls below a warp size. From there, we wait on all threads to
// ensure all threads have finished writing their contributions to shared
// memory. Once the threads are synced, we remove the control divergence in the
// final warp by doing a warp-level reduce. Since there are no branches in the
// warp-reduce and we do not go to shared or global memory, we greatly lower the
// control divergence and runtime performance associated with fewer memory
// accesses. Note that control divergence isn't fully removed since we have one
// final check to place the output value.
__global__ void
simple_shared_memory_reduction_kernel_warp_primitives(float *input, float *,
                                                      float *output, int N) {
  const int BLOCK_DIM = 128;
  // Safety check
  assert(BLOCK_DIM == blockDim.x);

  __shared__ float input_s[BLOCK_DIM];

  int i = threadIdx.x;
  input_s[i] = input[i] + input[i + blockDim.x];
  for (int stride = blockDim.x / 2; stride >= warpSize; stride /= 2) {
    __syncthreads();
    if (i < stride) {
      input_s[i] += input_s[i + stride];
    }
  }

  // Ensure all threads have written their shared memory values before having
  // the first warp reduce
  __syncthreads();

  if (warpIdx() == 0) {
    float total = warp_reduce(input_s[i]);
    if (i == 0) {
      *output = total;
    }
  }
}

inline void run_simple_shared_memory_reduction_kernel_warp_primitives(
    int n, dim3 threadsPerBlock, dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(simple_shared_memory_reduction_kernel_warp_primitives, dm.a, dm.b,
             dm.c, n, blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[0]);
}

// Further improvement to the above kernel via a two stage warp primitive
// reduction. First, the initial sum is computed from global memory and warp
// reduced by all the threads in a warp. All warps in the thread block are
// active throughout this process, each performing an independent reduction.
// This warp's partial sum contribution is written into one shared memory that
// is now NUM_WARPS big: BLOCK_DIM / 32. Each warp's thread 0 has the reduced
// value and that is written to shared memory. From there, the final reduction
// happens in warp 0 as before. This approach removes the shared memory accesses
// and barrier synchronization from the first stage in the previous kernel.
__global__ void two_stage_warp_primitives_reduction_kernel(float *input,
                                                           float *,
                                                           float *output,
                                                           int N) {
  const int BLOCK_DIM = 128;
  // Safety check
  assert(BLOCK_DIM == blockDim.x);
  assert(warpSize == 32);

  int i = threadIdx.x;
  float partial_sum = warp_reduce(input[i] + input[i + blockDim.x]);

  __shared__ float input_s[BLOCK_DIM / 32];
  // If I am the first thread in the warp, I have the partial sum contribution:
  // write it out to shared memory
  if (laneIdx() == 0) {
    input_s[warpIdx()] = partial_sum;
  }
  // Wait for all threads to contribute to shared mem
  __syncthreads();

  // The final reduction. input_s only has BLOCK_DIM/32 valid entries (one
  // per warp), so lanes beyond that must contribute 0 rather than reading
  // out of bounds.
  if (warpIdx() == 0) {
    float val = i < BLOCK_DIM / warpSize ? input_s[i] : 0.0f;
    float total = warp_reduce(val);
    if (i == 0) {
      *output = total;
    }
  }
}

inline void run_two_stage_warp_primitives_reduction_kernel(int n,
                                                           dim3 threadsPerBlock,
                                                           dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(two_stage_warp_primitives_reduction_kernel, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[0]);
}

// Up until now, we have been working with input sizes that fit within one
// block. This is mainly because we are using __syncthreads() as our
// synchronization mechanism. Note that this builtin only works on threads
// within the same block. To generalize reduction, split the input into segments
// that are of appropriate size for a block. Each block then performs its own
// reduction tree, and write its contribution using an atomic add operation.
// Alternatives to an atomic add:
// 1. Privatization: Each thread writes to a partial sum array using its index
// and another kernel merges the outputs into a final result.
// 2. Copy the partial sum array to host and have it do the reduction.
__global__ void general_reduce_kernel(float *input, float *, float *output,
                                      int N) {
  const int BLOCK_DIM = 128;
  // Safety check
  assert(BLOCK_DIM == blockDim.x);
  assert(warpSize == 32);

  int segment = 2 * blockDim.x * blockIdx.x;
  int i = segment + threadIdx.x;
  float partial_sum = warp_reduce(input[i] + input[i + blockDim.x]);

  __shared__ float input_s[BLOCK_DIM / 32];
  // First thread contributes in the warp.
  if (threadIdx.x % warpSize == 0) {
    input_s[threadIdx.x / warpSize] = partial_sum;
  }

  // Wait for all threads to contribute
  __syncthreads();

  // The final reduction
  if (warpIdx() == 0) {
    float val =
        threadIdx.x < BLOCK_DIM / warpSize ? input_s[threadIdx.x] : 0.0f;
    float total = warp_reduce(val);
    if (threadIdx.x == 0) {
      atomicAdd(output, total);
    }
  }
}

inline void run_general_reduce_kernel(int n, dim3 threadsPerBlock,
                                      dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);

  RUN_KERNEL(general_reduce_kernel, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  printf("%f\n", hm.c[0]);
}

int main() {
  print_cuda_properties();
  {
    int n = 256;
    dim3 threadsPerBlock(n / 2);
    dim3 blocksPerGrid(1);
    run_simple_reduction_kernel(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 256;
    dim3 threadsPerBlock(n / 2);
    dim3 blocksPerGrid(1);
    run_simple_reduction_kernel_decreasing_stride(n, threadsPerBlock,
                                                  blocksPerGrid);
  }
  {
    int n = 256;
    dim3 threadsPerBlock(n / 2);
    dim3 blocksPerGrid(1);
    run_simple_shared_memory_reduction_kernel(n, threadsPerBlock,
                                              blocksPerGrid);
  }
  {
    int n = 256;
    dim3 threadsPerBlock(n / 2);
    dim3 blocksPerGrid(1);
    run_simple_shared_memory_reduction_kernel_warp_primitives(
        n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 256;
    dim3 threadsPerBlock(n / 2);
    dim3 blocksPerGrid(1);
    run_two_stage_warp_primitives_reduction_kernel(n, threadsPerBlock,
                                                   blocksPerGrid);
  }
  {
    int n = 2048; // purposely more data than we can fit into a block
    dim3 threadsPerBlock(128);
    // Each thread reduces 2 elements (input[i] + input[i + blockDim.x]), so
    // each block covers 2 * threadsPerBlock.x elements.
    dim3 blocksPerGrid(ceildiv(n, 2 * threadsPerBlock.x));
    run_general_reduce_kernel(n, threadsPerBlock, blocksPerGrid);
  }
}
