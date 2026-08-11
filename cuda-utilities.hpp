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
    host_memory hm;

    const auto bytes = n * sizeof(float);

    hosts.emplace_back(static_cast<float *>(malloc(bytes)));
    hm.a = hosts.back().get();

    hosts.emplace_back(static_cast<float *>(malloc(bytes)));
    hm.b = hosts.back().get();

    hosts.emplace_back(static_cast<float *>(malloc(bytes)));
    hm.c = hosts.back().get();

    hm.n = bytes;

    return hm;
  }

  [[nodiscard]] auto upload_inputs_to_device(host_memory hm) {
    device_memory dm;
    dm.n = hm.n;

    CUDA_CHECK(cudaMalloc(&dm.a, dm.n));
    devices.emplace_back(device_ptr_t(dm.a, safe_cuda_free));
    dm.a = devices.back().get();

    CUDA_CHECK(cudaMalloc(&dm.b, dm.n));
    devices.emplace_back(device_ptr_t(dm.b, safe_cuda_free));
    dm.b = devices.back().get();

    CUDA_CHECK(cudaMalloc(&dm.c, dm.n));
    devices.emplace_back(device_ptr_t(dm.c, safe_cuda_free));
    dm.c = devices.back().get();

    CUDA_CHECK(cudaMemcpy(dm.a, hm.a, hm.n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.b, hm.b, hm.n, cudaMemcpyHostToDevice));

    return dm;
  }

  [[nodiscard]] auto download_result_to_host(device_memory dm, host_memory hm) {
    CUDA_CHECK(cudaMemcpy(hm.c, dm.c, dm.n, cudaMemcpyDeviceToHost));
  }

private:
  std::vector<device_ptr_t> devices;
  std::vector<host_ptr_t> hosts;
};

#define RUN_KERNEL_MAIN(BYTE_COUNT, kern, N, gDim, bDim)                       \
  do {                                                                         \
    cuda_context ctx;                                                          \
                                                                               \
    auto hm = ctx.allocate_host_memory(BYTE_COUNT);                            \
    for (int i = 0; i < BYTE_COUNT; i++) {                                     \
      hm.a[i] = i * 1.0f;                                                      \
      hm.b[i] = i * 2.0f;                                                      \
    }                                                                          \
                                                                               \
    auto dm = ctx.upload_inputs_to_device(hm);                                 \
    RUN_KERNEL(kern, dm.a, dm.b, dm.c, N, gDim, bDim);                         \
    ctx.download_result_to_host(dm, hm);                                       \
                                                                               \
    printf("%f\n", hm.c[BYTE_COUNT - 1]);                                      \
  } while (0)
