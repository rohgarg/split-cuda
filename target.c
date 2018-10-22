#include <stdio.h>
#include <unistd.h>

#include <cuda.h>
#include "upper-half-cuda-wrappers.h"

static void processArgs(int, const char** );

int
main(int argc, char **argv)
{
  int i = 0;

  processArgs(argc, (const char**)argv);

  printf("cuInit returned: %d\n", (int)cuInit(0));

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
