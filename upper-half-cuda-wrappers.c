#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "upper-half-wrappers.h"
#include "upper-half-cuda-wrappers.h"

#undef cuInit
#undef cudaMalloc

CUresult
cuInit(unsigned int Flags)
{
  CUresult rc = 0;
  static __typeof__(&cuInit) cuInitFnc = (__typeof__(&cuInit)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (cuInitFnc == (__typeof__(&cuInit)) - 1) {
    LhDlsym_t dlsymFptr = (LhDlsym_t)lhInfo.lhDlsym;
    cuInitFnc = (__typeof__(&cuInit))dlsymFptr(Cuda_Fnc_cuInit);
  }
  // XXX: Don't need to change the fs register for the dlsym call
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  rc = cuInitFnc(Flags);
  RETURN_TO_UPPER_HALF();
  return rc;
}

cudaError_t
cudaMalloc(void **pointer, size_t size)
{
  cudaError_t rc = 0;
  DLOG(NOISE, "cudaMalloc(%p,%zu)\n", *pointer, size);
  static __typeof__(&cudaMalloc) cudaMallocFnc = (__typeof__(&cudaMalloc)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (cudaMallocFnc == (__typeof__(&cudaMalloc)) - 1) {
    LhDlsym_t dlsymFptr = (LhDlsym_t)lhInfo.lhDlsym;
    cudaMallocFnc = (__typeof__(&cudaMalloc))dlsymFptr(Cuda_Fnc_cudaMalloc);
  }
  // XXX: Don't need to change the fs register for the dlsym call
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  rc = cudaMallocFnc(pointer, size);
  RETURN_TO_UPPER_HALF();
  return rc;
}