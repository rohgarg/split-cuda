#define _GNU_SOURCE // For MAP_ANONYMOUS
#include <errno.h>
#include <stddef.h>
#include <sys/mman.h>

#include "common.h"
#include "kernel-loader.h"

#define MMAP_OFF_HIGH_MASK ((-(4096ULL << 1) << (8 * sizeof (off_t) - 1)))
#define MMAP_OFF_LOW_MASK  (4096ULL - 1)
#define MMAP_OFF_MASK (MMAP_OFF_HIGH_MASK | MMAP_OFF_LOW_MASK)
#define _real_mmap mmap

// TODO:
//  1. Make the size of list dynamic
//  2. Remove region from list when the application calls munmap
#define MAX_TRACK   1000
static int numRegions = 0;
static MmapInfo_t mmaps[MAX_TRACK] = {0};

// Returns a pointer to the array of mmap-ed regions
// Sets num to the number of valid items in the array
MmapInfo_t*
getMmappedList(int *num)
{
  if (!num) return NULL;
  *num = numRegions;
  return mmaps;
}

void*
mmapWrapper(void *addr, size_t length, int prot,
            int flags, int fd, off_t offset)
{
  void *ret = MAP_FAILED;
  if (offset & MMAP_OFF_MASK) {
    errno = EINVAL;
    return ret;
  }
  ret = _real_mmap(addr, length, prot, flags, fd, offset);
  if (ret != MAP_FAILED) {
    mmaps[numRegions].addr = ret;
    mmaps[numRegions].len = length;
    numRegions = (numRegions + 1) % MAX_TRACK;
    DLOG(NOISE, "LH: mmap (%d): %p @ %zu\n", numRegions, ret, length);
  }
  return ret;
}
