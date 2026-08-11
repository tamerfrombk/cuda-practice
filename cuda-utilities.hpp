#pragma once

#include <cstdlib>

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
#define RUN_KERNEL(kern, A, B, C, N, gDim, bDim) \
  do { \
    kern<<<(gDim), (bDim)>>>((A), (B), (C), (N)); \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)

constexpr auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };