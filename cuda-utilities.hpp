#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Launches are async and report nothing through <<<>>>, so check explicitly:
// cudaGetLastError catches launch-time failures (bad config, no kernel image
// for this GPU), cudaDeviceSynchronize catches faults during execution.
#define RUN_KERNEL(kern, A, B, C, N, gDim, bDim)                               \
  do {                                                                         \
    printf("KERN=%s N=%d gridDim(%d, %d, %d) blockDim(%d, %d, %d)\n", #kern,   \
           N, gDim.x, gDim.y, gDim.z, bDim.x, bDim.y, bDim.z);                 \
    kern<<<(gDim), (bDim)>>>((A), (B), (C), (N));                              \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

constexpr auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

inline auto print_cuda_properties = [] {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    printf("dev:%d:%s | maxThreadsPerBlock=%d multiProcessorCount=%d, "
           "clockRate=%d, maxThreadsDim(%d, %d, %d), maxGridSize(%d, %d, %d), "
           "regsPerBlock=%d, warpSize=%d\n",
           i, prop.name, prop.maxThreadsPerBlock, prop.multiProcessorCount,
           prop.clockRate, prop.maxThreadsDim[0], prop.maxThreadsDim[1],
           prop.maxThreadsDim[2], prop.maxGridSize[0], prop.maxGridSize[1],
           prop.maxGridSize[2], prop.regsPerBlock, prop.warpSize);
  }
};

struct host_memory {
  float *a, *b, *c;
  std::size_t n;
};

struct device_memory {
  float *a, *b, *c;
  std::size_t n;
};

inline void safe_cuda_free(void *p) { CUDA_CHECK(cudaFree(p)); }

class cuda_context {
  using device_ptr_t = std::unique_ptr<float[], decltype(&safe_cuda_free)>;
  using host_ptr_t = std::unique_ptr<float[]>;

public:
  [[nodiscard]] auto allocate_host_memory(size_t n) {
    const auto byte_count = n * sizeof(float);

    float *ps[3];
    for (int i = 0; i < 3; ++i) {
      hosts.emplace_back(static_cast<float *>(malloc(byte_count)));
      ps[i] = hosts.back().get();
    }

    return host_memory{
        .a = ps[0],
        .b = ps[1],
        .c = ps[2],
        .n = byte_count,
    };
  }

  [[nodiscard]] auto upload_inputs_to_device(host_memory hm) {
    float *ps[3];
    for (int i = 0; i < 3; ++i) {
      CUDA_CHECK(cudaMalloc(&ps[i], hm.n));
      devices.emplace_back(ps[i], safe_cuda_free);
    }

    CUDA_CHECK(cudaMemcpy(ps[0], hm.a, hm.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ps[1], hm.b, hm.n, cudaMemcpyHostToDevice));

    return device_memory{
        .a = ps[0],
        .b = ps[1],
        .c = ps[2],
        .n = hm.n,
    };
  }

  [[nodiscard]] auto download_result_to_host(device_memory dm, host_memory hm) {
    CUDA_CHECK(cudaMemcpy(hm.c, dm.c, dm.n, cudaMemcpyDeviceToHost));
  }

private:
  std::vector<device_ptr_t> devices;
  std::vector<host_ptr_t> hosts;
};
