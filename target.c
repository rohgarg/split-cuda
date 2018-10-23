#include <stdio.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "upper-half-cuda-wrappers.h"

static void processArgs(int, const char** );

int
main(int argc, char **argv)
{
  int i = 0;
  void *cuda_ptr = NULL;

  processArgs(argc, (const char**)argv);

  // Allocate memory on GPU device for 1 integer
  cudaError_t rc = cudaMalloc(&cuda_ptr, sizeof(int));
  printf("cudaMalloc returned: %d, cuda_ptr: %p\n", (int)rc, cuda_ptr);

  while (1) {
    printf("%d ", i);
    fflush(stdout);
    sleep(1);
    i++;
  }
  return 0;
}

static void
processArgs(int argc, const char** argv)
{
  if (argc > 1) {
    printf("Application was called with the following args: ");
    for (int j = 1; j < argc; j++) {
      printf("%s ", argv[j]);
    }
    printf("\n");
  }
}
