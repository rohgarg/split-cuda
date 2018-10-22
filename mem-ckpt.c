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
#include "ckpt-restart.h"
#include "procmapsutils.h"
#include "upper-half-wrappers.h"
#include "utils.h"

const char *PROC_SELF_MAPS = "/proc/self/maps";
const char *CKPT_IMG = "./ckpt.img";

static CkptOrRestore_t state = CKPT;

static void checkpointHandler(int , siginfo_t *, void *);
static int skipRegion(const Area *);
static void checkpointMemory(int );
static ssize_t writeMemoryRegion(int , const Area *);
static void getSp(void **sp);
static void getFs(void *fs);
static void checkpointContext(int , const CkptRestartState_t *);

__attribute__ ((constructor))
void
installCkptHandler()
{
  struct sigaction sig_action;
  memset(&sig_action, 0, sizeof(sig_action));
  sig_action.sa_sigaction = checkpointHandler;
  sig_action.sa_flags = SA_RESTART | SA_SIGINFO;
  sigemptyset(&sig_action.sa_mask);
  int rc = sigaction(CKPT_SIGNAL, &sig_action, 0);
  if (rc < 0) {
    DLOG(ERROR, "Failed to install checkpoint signal handler. Error: %s. "
         "Exiting...\n", strerror(errno));
    exit(-1);
  }
}

// Local functions
static void
checkpointHandler(int signal, siginfo_t *info, void *ctx)
{
  CkptRestartState_t st = {0};
  int ckptfd = open(CKPT_IMG, O_WRONLY | O_CREAT, 0644);
  if (ckptfd < 0) {
    DLOG(ERROR, "Failed to open ckpt image for saving state. Error: %s\n",
         strerror(errno));
    return;
  }
  int rc = getcontext(&st.ctx);
  if (rc < 0) {
    DLOG(ERROR, "Failed to get context for the process. Error: %s\n",
         strerror(errno));
    return;
  }
  if (state == CKPT) {
    // Change the state before copying memory
    state = RESTORE;
    getSp(&st.sp);
    getFs(&st.fsAddr);
    checkpointContext(ckptfd, &st);
    checkpointMemory(ckptfd);
  } else {
   // We are running the restart code
   state = CKPT; // Reset state again for subsequent checkpoints
   reset_wrappers();
   return;
  }
}

static void
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

  if (strstr(area->name, "vvar") ||
      strstr(area->name, "vdso") ||
      strstr(area->name, "vsyscall"))
    return 1;

  // Don't skip the regions mmaped by the upper half
  if (lhInfo.lhMmapListFptr) {
    GetMmappedListFptr_t fnc = (GetMmappedListFptr_t) lhInfo.lhMmapListFptr;
    int numUhRegions = 0;
    MmapInfo_t *array = fnc(&numUhRegions);
    for (int i = 0; i < numUhRegions; i++) {
      // FIXME: Either the start addresses should match, or the end addresses
      // should match. The problem is that the upper half might have split one
      // large memory region into multiple by calling mprotect on subregions
      // in the large region. We need to fix this.
      if (array[i].addr == area->addr ||
          ((uintptr_t)array[i].addr + array[i].len) <= (uintptr_t)area->endAddr)
      {
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

static void
getSp(void **sp)
{
#if defined(__i386__) || defined(__x86_64__)
  asm volatile (CLEAN_FOR_64_BIT(mov %%esp, %0)
                  : "=g" (*sp)
                    : : "memory");
#elif defined(__arm__) || defined(__aarch64__)
  asm volatile ("mov %0,sp"
                : "=r" (*sp)
                : : "memory");
#else // if defined(__i386__) || defined(__x86_64__)
# error "assembly instruction not translated"
#endif // if defined(__i386__) || defined(__x86_64__)
}

static void
getFs(void *fs)
{
  assert(fs);
  int rc = syscall(SYS_arch_prctl, ARCH_GET_FS, fs);
  if (rc < 0) {
    DLOG(ERROR, "Could not retrieve fs register value. Error: %s\n",
         strerror(errno));
  }
}

static void
checkpointContext(int ckptfd, const CkptRestartState_t *st)
{
  int rc = writeAll(ckptfd, st, sizeof *st);
  if (rc < sizeof *st) {
    DLOG(ERROR, "Failed to write out ckpt-restart state. Error: %s\n",
         strerror(errno));
  }
}
