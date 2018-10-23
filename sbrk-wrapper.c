#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

#include "common.h"
#include "kernel-loader.h"

static void *__curbrk;
static void *__endOfHeap = 0;

void
setEndOfHeap(void *addr)
{
  __curbrk = addr;
  __endOfHeap = (void*)ROUND_UP(__curbrk);
}


/* Extend the process's data space by INCREMENT.
   If INCREMENT is negative, shrink data space by - INCREMENT.
   Return start of new space allocated, or -1 for errors.  */
void*
sbrkWrapper(intptr_t increment)
{
  void *oldbrk;

  DLOG(NOISE, "LH: sbrk called with 0x%lx\n", increment);

  if (__curbrk == NULL) {
    if (brk (0) < 0) {
      return (void *) -1;
    } else {
      __endOfHeap = __curbrk;
    }
  }

  if (increment == 0) {
    DLOG(NOISE, "LH: sbrk returning %p\n", __curbrk);
    return __curbrk;
  }

  oldbrk = __curbrk;
  if (increment > 0
      ? ((uintptr_t) oldbrk + (uintptr_t) increment < (uintptr_t) oldbrk)
      : ((uintptr_t) oldbrk < (uintptr_t) -increment))
    {
      errno = ENOMEM;
      return (void *) -1;
    }

  if ((VA)oldbrk + increment > (VA)__endOfHeap) {
    if (mmapWrapper(__endOfHeap,
                    ROUND_UP((VA)oldbrk + increment - (VA)__endOfHeap),
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS,
                    -1, 0) == MAP_FAILED) {
       return (void *) -1;
    }
  }

  __endOfHeap = (void*)ROUND_UP((VA)oldbrk + increment);
  __curbrk = (VA)oldbrk + increment;

  DLOG(NOISE, "LH: sbrk returning %p\n", oldbrk);

  return oldbrk;
}
