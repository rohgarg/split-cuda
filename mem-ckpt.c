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

static CkptOrRestore_t state = RUNNING;

static void checkpointHandler(int , siginfo_t *, void *);
static int skipRegion(const Area *);
static void checkpointMemory(int );
static ssize_t writeMemoryRegion(int , const Area *);
static void getSp(void **sp);
static void getFs(void *fs);
static void checkpointContext(int , const CkptRestartState_t *);
static ssize_t writeRelevantMemoryRegion(int , const Area *);

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
  state = RUNNING;
}

#undef doCheckpoint

// Public API for application-initiated checkpointing

CkptOrRestore_t
doCheckpoint()
{
  checkpointHandler(12, NULL, NULL);
  CkptOrRestore_t tempState = state;
  if (tempState == POST_RESTART || tempState == POST_RESUME) {
   state = RUNNING; // Reset state again for subsequent checkpoints
  }
  return tempState;
}

// Local functions
static void
checkpointHandler(int signal, siginfo_t *info, void *ctx)
{
  CkptRestartState_t st = {0};
  int ckptfd = open(CKPT_IMG, O_WRONLY | O_CREAT | O_TRUNC, 0644);
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
  if (state == RUNNING || state == POST_RESUME) {
    // Change the state before copying memory
    state = POST_RESTART;
    getSp(&st.sp);
    getFs(&st.fsAddr);
    checkpointContext(ckptfd, &st);
    checkpointMemory(ckptfd);
    close(ckptfd);
    // Now, change the state back to resuming to inform the application
    state = POST_RESUME;
  } else if (state == POST_RESTART) {
   // We are running the restart code
   reset_wrappers();
  }
}

// Returns true if needle is in the haystack
static inline int
regionContains(const void *haystackStart,
               const void *haystackEnd,
               const void *needleStart,
               const void *needleEnd)
{
  return needleStart >= haystackStart && needleEnd <= haystackEnd;
}

static void
checkpointMemory(int ckptfd)
{
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  if(mapsfd == -1) {
    DLOG(ERROR, "Unable to open '%s'. Error %d.", PROC_SELF_MAPS, errno);
    exit(EXIT_FAILURE);
  }
  Area area;
  while (readMapsLine(mapsfd, &area)) {
    if(!skipRegion(&area)) {
      writeRelevantMemoryRegion(ckptfd, &area);
    }
  }
  close(mapsfd);
}

static ssize_t
writeRelevantMemoryRegion(int ckptfd, const Area *area)
{
  ssize_t rc = 0;

  // Cannot handle regions with no Read perms
  if (!(area->prot & PROT_READ)) {
    return rc;
  }

  // Don't skip the regions mmaped by the upper half
  if (lhInfo.lhMmapListFptr) {
    GetMmappedListFptr_t fnc = (GetMmappedListFptr_t) lhInfo.lhMmapListFptr;
    int numUhRegions = 0;
    MmapInfo_t *array = fnc(&numUhRegions);
    for (int i = 0; i < numUhRegions; i++) {
      void *uhMmapStart = array[i].addr;
      void *uhMmapEnd = (VA)array[i].addr + array[i].len;
      if (regionContains(uhMmapStart, uhMmapEnd, area->addr, area->endAddr)) {
        rc = writeMemoryRegion(ckptfd, area);
        break;
      } else if (regionContains(area->addr, area->endAddr,
                                uhMmapStart, uhMmapEnd)) {
        Area uhArea = *area;
        uhArea.addr = (VA)uhMmapStart;
        uhArea.endAddr = (VA)uhMmapEnd;
        uhArea.size = array[i].len;
        rc = writeMemoryRegion(ckptfd, &uhArea);
        break;
      }
    }
  }
  return rc;
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

  return 0;
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
