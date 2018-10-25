#include <stdio.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "ckpt-restart.h"
#include "upper-half-cuda-wrappers.h"

static void processArgs(int, const char** );

int
main(int argc, char **argv)
{
  int i = 0;
  void *cuda_ptr1 = NULL;
  void *cuda_ptr2 = NULL;

  processArgs(argc, (const char**)argv);

  // Allocate memory on GPU device for 1 integer
  cudaError_t rc = cudaMalloc(&cuda_ptr1, sizeof(int));
  printf("cudaMalloc returned: %d, cuda_ptr1: %p\n", (int)rc, cuda_ptr1);

  while (i < 2) {
    printf("%d ", i);
    fflush(stdout);
    sleep(2);
    i++;
  }
  CkptOrRestore_t ret = doCheckpoint();
  if (ret == POST_RESUME) {
    printf("\nResuming after ckpt...\n");
  } else if (ret == POST_RESTART) {
    printf("\nRestarting from a ckpt...\n");
  }

  rc = cudaMalloc(&cuda_ptr2, sizeof(int));
  printf("\ncudaMalloc returned: %d, cuda_ptr2: %p\n", (int)rc, cuda_ptr2);

  while (i < 4) {
    printf("%d ", i);
    fflush(stdout);
    sleep(2);
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
