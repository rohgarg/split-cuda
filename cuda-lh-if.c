#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

static void* Cuda_Fnc_Ptrs[] = {
  NULL,
  FOREACH_FNC(GENERATE_FNC_PTR)
  NULL,
};

void*
lhDlsym(Cuda_Fncs_t fnc)
{
  DLOG(INFO, "LH: Dlsym called with: %d\n", fnc);
  if (fnc < Cuda_Fnc_NULL || fnc > Cuda_Fnc_Invalid) {
    return NULL;
  }
  void *addr = Cuda_Fnc_Ptrs[fnc];
  return addr;
}
