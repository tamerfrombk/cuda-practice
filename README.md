# CUDA Practice

CUDA kernels and exercises worked through from
[Programming Massively Parallel Processors](https://www.amazon.com/dp/0443439001)
(Hwu, Kirk & El Hajj) for GPU performance-engineering skills.

## Environment
- GPU: GTX 1080
- CUDA Toolkit: 11.0 

## Kernels

| Exercise        | File           | Concepts                                             |
|-----------------|----------------|------------------------------------------------------|
| Vector addition | `vector-add.cu`| kernel launch, grid/block indexing, memory coalescing| 

## Build & run
```
nvcc vector-add.cu -o vector-add && ./vector-add
```
