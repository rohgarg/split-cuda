#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "common.h"
#include "upper-half-wrappers.h"

int initialized = 0;

static void readLhInfoAddr();

LowerHalfInfo_t lhInfo = {0};

void*
sbrk(intptr_t increment)
{
  static __typeof__(&sbrk) lowerHalfSbrkWrapper = (__typeof__(&sbrk)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfSbrkWrapper == (__typeof__(&sbrk)) - 1) {
    lowerHalfSbrkWrapper = (__typeof__(&sbrk))lhInfo.lhSbrk;
  }
  return lowerHalfSbrkWrapper(increment);
}

void*
mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
  static __typeof__(&mmap) lowerHalfMmapWrapper = (__typeof__(&mmap)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfMmapWrapper == (__typeof__(&mmap)) - 1) {
    lowerHalfMmapWrapper = (__typeof__(&mmap))lhInfo.lhMmap;
  }
  return lowerHalfMmapWrapper(addr, length, prot, flags, fd, offset);
}

void
initialize_wrappers()
{
  if (!initialized) {
    readLhInfoAddr();
    initialized = 1;
  }
}

static void
readLhInfoAddr()
{
  LowerHalfInfo_t *lhInfoPtr = NULL;
  int fd = open(LH_FILE_NAME, O_RDONLY);
  if (fd < 0) {
    DLOG(ERROR, "Could not open addr.bin for reading. Error: %s",
         strerror(errno));
    exit(-1);
  }

  int rc = read(fd, &lhInfo, sizeof(lhInfo));
  if (rc < sizeof(lhInfo)) {
    DLOG(ERROR, "Read fewer bytes than expected from addr.bin. Error: %s",
         strerror(errno));
    exit(-1);
  }
  unlink(LH_FILE_NAME);
  close(fd);
}

void
reset_wrappers()
{
  initialized = 0;
}
