#ifndef UPPER_HALF_CUDA_WRAPPERS_H
#define UPPER_HALF_CUDA_WRAPPERS_H

extern CUresult cuInit(unsigned int Flags) __attribute__((weak));
#define cuInit(f) (cuInit ? cuInit(f) : 0)

cudaError_t cudaMalloc(void **pointer, size_t size) __attribute__((weak));
#define cudaMalloc(f, g) (cudaMalloc ? cudaMalloc(f, g) : 0)


#endif // ifndef UPPER_HALF_CUDA_WRAPPERS_H