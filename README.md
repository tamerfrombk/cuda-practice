# CUDA Practice

CUDA kernels and exercises worked through from
[Programming Massively Parallel Processors](https://www.amazon.com/dp/0443439001)
(Hwu, Kirk & El Hajj) for GPU performance-engineering skills.

## Environment
- OS: Fedora 44
- GPU: GeForce GT 1030
- CUDA Toolkit: 12.9, invoked through the `nvcc12` wrapper
- Driver: 580.173

NOTE: Fedora 44 ships with CUDA 13 and latest 590+ drivers which are incompatible with my older generation GPU.
To compensate, CUDA Toolkit 12.9 is installed all programs must be built with `nvcc12` wrapper.

## Kernels

| Exercise        | File           | Concepts                                             |
|-----------------|----------------|------------------------------------------------------|
| Vector addition | `vector-add.cu`| kernel launch, grid/block indexing, memory coalescing| 
| Vector addition using float4 | `vector-add4.cu`| same as above but with float4 memory addressing| 
| Square MatMul | `simple-matrix-multiply.cu`| 2D grid/block indexing and memory coalescing| 
| Tiled MatMul | `tiled-matrix-multiply.cu`| Shared Memory and Tiling| 
| Histogram | `histogram.cu`| Atomics, Privatization, Thread Coarsening| 

## Build & run

```
cmake -B build
cmake --build build -t run-vector-add
```

`vector-add` can be replaced with any other file in the repo (as `run-<name>`) and the command is similar.
