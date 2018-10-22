#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "common.h"
#include "procmapsutils.h"
#include "utils.h"

const char *PROC_SELF_MAPS = "/proc/self/maps";

static int skipRegion(const Area *);
static ssize_t writeMemoryRegion(int , const Area *);

void
checkpointMemory(int ckptfd)
{
  Area area;
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  if(mapsfd == -1) {
    DLOG(ERROR, "Unable to open '%s'. Error %d.", PROC_SELF_MAPS, errno);
    exit(EXIT_FAILURE);
  }
  while (readMapsLine(mapsfd, &area)) {
    if(!skipRegion(&area)) {
      writeMemoryRegion(ckptfd, &area);
    }
  }
  close(mapsfd);
}

// Returns 1 for a region to be skipped, 0 otherwise
static int
skipRegion(const Area *area)
{
  if (!(area->prot & PROT_READ))
    return 1;

  if (!(strstr(area->name, "vvar") ||
        strstr(area->name, "vdso") ||
        strstr(area->name, "vsyscall")))
    return 1;

  // Don't skip the regions mmaped by the upper half
  if (lhInfo.lhMmapListFptr) {
    GetMmappedListFptr_t fnc = (GetMmappedListFptr_t) lhInfo.lhMmapListFptr;
    int numUhRegions = 0;
    void **array = fnc(&numUhRegions);
    for (int i = 0; i < numUhRegions; i++) {
      if (array[i] == area->addr) {
        return 0;
      }
    }
  }
  return 1;
}

static ssize_t
writeMemoryRegion(int fd, const Area *area)
{
  int rc = 0;
  rc += writeAll(fd, area, sizeof *area);
  rc += writeAll(fd, area->addr, area->size);
  return rc;
}
