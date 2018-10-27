#define _GNU_SOURCE // For MAP_ANONYMOUS
#include <errno.h>
#include <stddef.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include "common.h"
#include "kernel-loader.h"
#include "utils.h"
#include "trampoline_setup.h"

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

static void* __mmapWrapper(void* , size_t , int , int , int , off_t );
static void patchLibc(int , const void* , const char* );

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
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  ret = __mmapWrapper(addr, length, prot, flags, fd, offset);
  RETURN_TO_UPPER_HALF();
  return ret;
}

static void*
__mmapWrapper(void *addr, size_t length, int prot,
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
    if (fd > 0) {
      char glibcFullPath[PATH_MAX] = {0};
      int found = checkLibrary(fd, "libc-", glibcFullPath, PATH_MAX);
      if (found && (prot & PROT_EXEC)) {
        patchLibc(fd, ret, glibcFullPath);
      }
    }
  }
  return ret;
}

static void
patchLibc(int fd, const void *base, const char *glibc)
{
  assert(base);
  assert(fd > 0);
  const char *MMAP_SYMBOL_NAME = "mmap";
  const char *SBRK_SYMBOL_NAME = "sbrk";
  DLOG(INFO, "Patching libc (%s) @ %p\n", glibc, base);
  // Save incoming offset
  off_t saveOffset = lseek(fd, 0, SEEK_CUR);
  off_t mmapOffset = get_symbol_offset(fd, glibc, MMAP_SYMBOL_NAME);
  insertTrampoline((VA)base + mmapOffset, &mmapWrapper);
  off_t sbrkOffset = get_symbol_offset(fd, glibc, SBRK_SYMBOL_NAME);
  insertTrampoline((VA)base + sbrkOffset, &sbrkWrapper);
  DLOG(INFO, "Patched libc (%s) @ %p: offset(sbrk): %zx; offset(mmap): %zx\n",
       glibc, base, sbrkOffset, mmapOffset);
  // Restore file offset to not upset the caller
  lseek(fd, saveOffset, SEEK_SET);
}