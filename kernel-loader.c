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

#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "common.h"
#include "ckpt-restart.h"
#include "custom-loader.h"
#include "kernel-loader.h"
#include "procmapsutils.h"

LowerHalfInfo_t lhInfo;

// Local function declarations
static void getProcStatField(enum Procstat_t , char *, size_t );
static void getStackRegion(Area *);
static void* deepCopyStack(void *, const void *, size_t,
                           const void *, const void*,
                           const DynObjInfo_t *);
static void* createNewStackForRtld(const DynObjInfo_t *);
static void* createNewHeapForRtld(const DynObjInfo_t *);
static int insertTrampoline(void* , void* );
static void* getEntryPoint(DynObjInfo_t );
static void patchAuxv(ElfW(auxv_t) *, unsigned long ,
                      unsigned long , unsigned long );
static int writeLhInfoToFile();
static int setupLowerHalfInfo();
static void printUsage();
static void printRestartUsage();

// Global functions

// This function loads in ld.so, sets up a separate stack for it, and jumps
// to the entry point of ld.so
void
runRtld()
{
  int rc = -1;

  // Pointer to the ld.so entry point
  void *ldso_entrypoint = NULL;

  // Load RTLD (ld.so)
  char *ldname  = getenv("TARGET_LD");
  char *uhpreload = getenv("UH_PRELOAD");
  if (!ldname || !uhpreload) {
    printUsage();
    return;
  }

  DynObjInfo_t ldso = safeLoadLib(ldname);
  if (ldso.baseAddr == NULL || ldso.entryPoint == NULL) {
    DLOG(ERROR, "Error loading the runtime loader (%s). Exiting...\n", ldname);
    return;
  }

  DLOG(INFO, "New ld.so loaded at: %p\n", ldso.baseAddr);
  ldso_entrypoint = getEntryPoint(ldso);

  // Create new stack region to be used by RTLD
  void *newStack = createNewStackForRtld(&ldso);
  if (!newStack) {
    DLOG(ERROR, "Error creating new stack for RTLD. Exiting...\n");
    exit(-1);
  }
  DLOG(INFO, "New stack start at: %p\n", newStack);

  // Create new heap region to be used by RTLD
  void *newHeap = createNewHeapForRtld(&ldso);
  if (!newHeap) {
    DLOG(ERROR, "Error creating new heap for RTLD. Exiting...\n");
    exit(-1);
  }
  DLOG(INFO, "New heap mapped at: %p\n", newHeap);

  setEndOfHeap((VA)newHeap + PAGE_SIZE);
  rc = insertTrampoline(ldso.mmapAddr, &mmapWrapper);
  if (rc < 0) {
    DLOG(ERROR, "Error inserting trampoline for mmap. Exiting...\n");
    exit(-1);
  }
  rc = insertTrampoline(ldso.sbrkAddr, &sbrkWrapper);
  if (rc < 0) {
    DLOG(ERROR, "Error inserting trampoline for sbrk. Exiting...\n");
    exit(-1);
  }

  // Everything is ready, let's set up the lower-half info struct for the upper
  // half to read from
  rc = setupLowerHalfInfo();
  if (rc < 0) {
    DLOG(ERROR, "Failed to set up lhinfo for the upper half. Exiting...\n");
    exit(-1);
  }

  // Change the stack pointer to point to the new stack and jump into ld.so
  // TODO: Clean up all the registers?
  asm volatile (CLEAN_FOR_64_BIT(mov %0, %%esp; )
                : : "g" (newStack) : "memory");
  asm volatile ("jmp *%0" : : "g" (ldso_entrypoint) : "memory");
}

#ifdef STANDALONE
int
main(int argc, char **argv)
{
  if (argc < 2) {
    printUsage();
    return -1;
  }
  if (strstr(argv[1], "--restore")) {
    if (argc < 3) {
      printRestartUsage();
      return -1;
    }
    int rc = setupLowerHalfInfo();
    if (rc < 0) {
      DLOG(ERROR, "Failed to set up lhinfo for the upper half. Exiting...\n");
      exit(-1);
    }
    restoreCheckpoint(argv[2]);
  }
  runRtld();
  return 0;
}
#endif

// Local functions

static void
printUsage()
{
  DLOG(ERROR, "Usage: UH_PRELOAD=/path/to/libupperhalfwrappers.so "
          "TARGET_LD=/path/to/ld.so ./kernel-loader "
          "<target-application> [application arguments ...]\n");
}

static void
printRestartUsage()
{
  DLOG(ERROR, "Usage: ./kernel-loader --restore /path/to/ckpt.img\n");
}

// Returns the /proc/self/stat entry in the out string (of length len)
static void
getProcStatField(enum Procstat_t type, char *out, size_t len)
{
  const char *procPath = "/proc/self/stat";
  char sbuf[1024] = {0};
  int field_counter = 0;
  char *field_str = NULL;
  int fd, num_read;

  fd = open(procPath, O_RDONLY);
  if (fd < 0) {
    DLOG(ERROR, "Failed to open %s. Error: %s\n", procPath, strerror(errno));
    return;
  }

  num_read = read(fd, sbuf, sizeof sbuf - 1);
  close(fd);
  if (num_read <= 0) return;
  sbuf[num_read] = '\0';

  field_str = strtok(sbuf, " ");
  while (field_str && field_counter != type) {
    field_str = strtok(NULL, " ");
    field_counter++;
  }

  if (field_str) {
    strncpy(out, field_str, len);
  } else {
    DLOG(ERROR, "Failed to parse %s.\n", procPath);
  }
}

// Returns the [stack] area by reading the proc maps
static void
getStackRegion(Area *stack) // OUT
{
  Area area;
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  while (readMapsLine(mapsfd, &area)) {
    if (strstr(area.name, "[stack]") && area.endAddr >= (VA)&area) {
      *stack = area;
      break;
    }
  }
  close(mapsfd);
}

// Given a pointer to aux vector, parses the aux vector, and patches the
// following three entries: AT_PHDR, AT_ENTRY, and AT_PHNUM
static void
patchAuxv(ElfW(auxv_t) *av, unsigned long phnum,
          unsigned long phdr, unsigned long entry)
{
  for (; av->a_type != AT_NULL; ++av) {
    switch (av->a_type) {
      case AT_PHNUM:
        av->a_un.a_val = phnum;
        break;
      case AT_PHDR:
        av->a_un.a_val = phdr;
        break;
      case AT_ENTRY:
        av->a_un.a_val = entry;
        break;
      case AT_RANDOM:
        DLOG(NOISE, "AT_RANDOM value: 0%lx\n", av->a_un.a_val);
        break;
      default:
        break;
    }
  }
}

// Creates a deep copy of the stack region pointed to be `origStack` at the
// location pointed to be `newStack`. Returns the start-of-stack pointer
// in the new stack region.
static void*
deepCopyStack(void *newStack, const void *origStack, size_t len,
              const void *newStackEnd, const void *origStackEnd,
              const DynObjInfo_t *info)
{
  // This function assumes that this env var is set.
  assert(getenv("TARGET_LD"));
  assert(getenv("UH_PRELOAD"));

  // Return early if any pointer is NULL
  if (!newStack || !origStack ||
      !newStackEnd || !origStackEnd ||
      !info) {
    return NULL;
  }

  // First, we do a shallow copy, which is essentially, just copying the
  // bits from the original stack into the new stack.
  memcpy(newStack, origStack, len);

  // Next, turn the shallow copy into a deep copy.
  //
  // The main thing we need to do is to patch the argv and env vectors in
  // the new stack to point to addresses in the new stack region. Note that
  // the argv and env are simply arrays of pointers. The pointers point to
  // strings in other locations in the stack.

  void *origArgcAddr     = (void*)GET_ARGC_ADDR(origStackEnd);
  int  origArgc          = *(int*)origArgcAddr;
  char **origArgv        = (void*)GET_ARGV_ADDR(origStackEnd);
  const char **origEnv   = (void*)GET_ENV_ADDR(origArgv, origArgc);

  void *newArgcAddr     = (void*)GET_ARGC_ADDR(newStackEnd);
  int  newArgc          = *(int*)newArgcAddr;
  char **newArgv        = (void*)GET_ARGV_ADDR(newStackEnd);
  const char **newEnv   = (void*)GET_ENV_ADDR(newArgv, newArgc);
  ElfW(auxv_t) *newAuxv = GET_AUXV_ADDR(newEnv);

  // Patch the argv vector in the new stack
  //   First, set up the argv vector based on the original stack
  for (int i = 0; origArgv[i] != NULL; i++) {
    off_t argvDelta = (uintptr_t)origArgv[i] - (uintptr_t)origArgv;
    newArgv[i] = (char*)((uintptr_t)newArgv + (uintptr_t)argvDelta);
  }

  //   Next, we patch argv[0], the first argument, on the new stack
  //   to point to "/path/to/ld.so".
  //
  //   From the point of view of ld.so, it would appear as if it was called
  //   like this: $ /lib/ld.so /path/to/target.exe app-args ...
  //
  //   NOTE: The kernel loader needs to be called with at least two arguments
  //   to get a stack that is 16-byte aligned at the start. Since we want to
  //   be able to jump into ld.so with at least two arguments (ld.so and the
  //   target exe) on the new stack, we also need two arguments on the
  //   original stack.
  //
  //   If the original stack had just one argument, we would have inherited
  //   that alignment in the new stack. Trying to push in another argument
  //   (target exe) on the new stack would destroy the 16-byte alignment
  //   on the new stack. This would lead to a crash later on in ld.so.
  //
  //   The problem is that there are instructions (like, "movaps") in ld.so's
  //   code that operate on the stack memory region and require their
  //   operands to be 16-byte aligned. A non-16-byte-aligned operand (for
  //   example, the stack base pointer) leads to a general protection
  //   exception (#GP), which translates into a segfault for the user
  //   process.
  //
  //   The Linux kernel ensures that the start of stack is always 16-byte
  //   aligned. It seems like this is part of the Linux kernel x86-64 ABI.
  //   For example, see here:
  //
  //     https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L150
  //
  //     https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L288
  //
  //   (The kernel uses the STACK_ROUND macro to first set up the stack base
  //    at a 16-byte aligned address, and then pushes items on the stack.)
  //
  //   We could do something similar on the new stack region. But perhaps it's
  //   easier to just depend on the original stack having at least two args:
  //   "/path/to/kernel-loader" and "/path/to/target.exe".
  //
  //   NOTE: We don't need to patch newArgc, since the original stack,
  //   from where we would have inherited the data in the new stack, already
  //   had the correct value for origArgc. We just make argv[0] in the
  //   new stack to point to "/path/to/ld.so", instead of
  //   "/path/to/kernel-loader".
  off_t argvDelta = (uintptr_t)getenv("TARGET_LD") - (uintptr_t)origArgv;
  newArgv[0] = (char*)((uintptr_t)newArgv + (uintptr_t)argvDelta);

  // Patch the env vector in the new stack
  for (int i = 0; origEnv[i] != NULL; i++) {
    off_t envDelta = (uintptr_t)origEnv[i] - (uintptr_t)origEnv;
    newEnv[i] = (char*)((uintptr_t)newEnv + (uintptr_t)envDelta);
  }

  // Change "UH_PRELOAD" to "LD_PRELOAD". This way, upper half's ld.so
  // will preload the upper half wrapper library.
  char **newEnvPtr = (char**)newEnv;
  for (; *newEnvPtr; newEnvPtr++) {
    if (strstr(*newEnvPtr, "UH_PRELOAD")) {
      (*newEnvPtr)[0] = 'L';
      (*newEnvPtr)[1] = 'D';
      break;
    }
  }

  // The aux vector, which we would have inherited from the original stack,
  // has entries that correspond to the kernel loader binary. In particular,
  // it has these entries AT_PHNUM, AT_PHDR, and AT_ENTRY that correspond
  // to kernel-loader. So, we atch the aux vector in the new stack to
  // correspond to the new binary: the freshly loaded ld.so.
  patchAuxv(newAuxv, info->phnum,
            (uintptr_t)info->phdr,
            (uintptr_t)info->entryPoint);

  // We clear out the rest of the new stack region just in case ...
  memset(newStack, 0, (void*)&newArgv[-2] - newStack);

  // Return the start of new stack.
  return (void*)newArgcAddr;
}

// This function does three things:
//  1. Creates a new stack region to be used for initialization of RTLD (ld.so)
//  2. Deep copies the original stack (from the kernel) in the new stack region
//  3. Returns a pointer to the beginning of stack in the new stack region
static void*
createNewStackForRtld(const DynObjInfo_t *info)
{
  Area stack;
  char stackEndStr[20] = {0};
  getStackRegion(&stack);

  // 1. Allocate new stack region
  // We go through the mmap wrapper function to ensure that this gets added
  // to the list of upper half regions to be checkpointed.
  void *newStack = mmapWrapper(NULL, stack.size, PROT_READ | PROT_WRITE,
                               MAP_GROWSDOWN | MAP_PRIVATE | MAP_ANONYMOUS,
                               -1, 0);
  if (newStack == MAP_FAILED) {
    DLOG(ERROR, "Failed to mmap new stack region: %s\n", strerror(errno));
    return NULL;
  }
  DLOG(INFO, "New stack mapped at: %p\n", newStack);

  // 3. Get pointer to the beginning of the stack in the new stack region
  // The idea here is to look at the beginning of stack in the original
  // stack region, and use that to index into the new memory region. The
  // same offsets are valid in both the stack regions.
  getProcStatField(STARTSTACK, stackEndStr, sizeof stackEndStr);

  // NOTE: The kernel sets up the stack in the following format.
  //      -1(%rsp)                       Stack end for application
  //      0(%rsp)                        argc (Stack start for application)
  //      LP_SIZE(%rsp)                  argv[0]
  //      (2*LP_SIZE)(%rsp)              argv[1]
  //      ...
  //      (LP_SIZE*(argc))(%rsp)         NULL
  //      (LP_SIZE*(argc+1))(%rsp)       envp[0]
  //      (LP_SIZE*(argc+2))(%rsp)       envp[1]
  //      ...
  //                                     NULL
  //
  // NOTE: proc-stat returns the address of argc on the stack.
  // argv[0] is 1 LP_SIZE ahead of argc, i.e., startStack + sizeof(void*)
  // Stack End is 1 LP_SIZE behind argc, i.e., startStack - sizeof(void*)
  // sizeof(unsigned long) == sizeof(void*) == 8 on x86-64
  unsigned long origStackEnd = atol(stackEndStr) - sizeof(unsigned long);
  unsigned long origStackOffset = origStackEnd - (unsigned long)stack.addr;
  unsigned long newStackOffset = origStackOffset;
  void *newStackEnd = (void*)((unsigned long)newStack + newStackOffset);

  // 2. Deep copy stack
  newStackEnd = deepCopyStack(newStack, stack.addr, stack.size,
                              (void*)newStackEnd, (void*)origStackEnd,
                              info);

  return newStackEnd;
}

// This function allocates a new heap for (the possibly second) ld.so.
// The initial heap size is 1 page
//
// Returns the start address of the new heap on success, or NULL on
// failure.
static void*
createNewHeapForRtld(const DynObjInfo_t *info)
{
  // We go through the mmap wrapper function to ensure that this gets added
  // to the list of upper half regions to be checkpointed.
  void *addr = mmapWrapper(0, 10*PAGE_SIZE, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    DLOG(ERROR, "Failed to mmap region. Error: %s\n",
         strerror(errno));
    return NULL;
  }
  return addr;
}

// Returns 0 on success, -1 on failure
static int
insertTrampoline(void *from_addr, void *to_addr)
{
  int rc;
#if defined(__x86_64__)
  unsigned char asm_jump[] = {
    // mov    $0x1234567812345678,%rax
    0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12,
    // jmpq   *%rax
    0xff, 0xe0
  };
  // Beginning of address in asm_jump:
  const int addr_offset = 2;
#elif defined(__i386__)
    static unsigned char asm_jump[] = {
      0xb8, 0x78, 0x56, 0x34, 0x12, // mov    $0x12345678,%eax
      0xff, 0xe0                    // jmp    *%eax
  };
  // Beginning of address in asm_jump:
  const int addr_offset = 1;
#else
# error "Architecture not supported"
#endif

  void *page_base = (void *)ROUND_DOWN(from_addr);
  int page_length = PAGE_SIZE;
  if ((VA)from_addr + sizeof(asm_jump) - (VA)page_base > PAGE_SIZE) {
    // The patching instructions cross page boundary. View page as double size.
    page_length = 2 * PAGE_SIZE;
  }

  // Temporarily add write permissions
  rc = mprotect(page_base, page_length, PROT_READ | PROT_WRITE | PROT_EXEC);
  if (rc < 0) {
    DLOG(ERROR, "mprotect failed: %s\n", strerror(errno));
    return -1;
  }

  // Now, do the patching
  memcpy(from_addr, asm_jump, sizeof(asm_jump));
  memcpy((VA)from_addr + addr_offset, (void*)&to_addr, sizeof(&to_addr));

  // Finally, remove the write permissions
  rc = mprotect(page_base, page_length, PROT_READ | PROT_EXEC);
  if (rc < 0) {
    DLOG(ERROR, "mprotect failed: %s\n", strerror(errno));
    return -1;
  }
  return rc;
}

// This function returns the entry point of the ld.so executable given
// the library handle
static void*
getEntryPoint(DynObjInfo_t info)
{
  return info.entryPoint;
}

// Writes out the lhinfo global object to a file. Returns 0 on success,
// -1 on failure.
static int
writeLhInfoToFile()
{
  int rc = 0;

  int fd = open(LH_FILE_NAME, O_WRONLY | O_CREAT, 0644);
  if (fd < 0) {
    DLOG(ERROR, "Could not create addr.bin file. Error: %s", strerror(errno));
    return -1;
  }

  rc = write(fd, &lhInfo, sizeof(lhInfo));
  if (rc < sizeof(lhInfo)) {
    DLOG(ERROR, "Wrote fewer bytes than expected to addr.bin. Error: %s",
         strerror(errno));
    rc = -1;
  }
  close(fd);
  return rc;
}

// Sets up lower-half info struct for the upper half to read from. Returns 0
// on success, -1 otherwise
static int
setupLowerHalfInfo()
{
  lhInfo.lhSbrk = &sbrkWrapper;
  lhInfo.lhMmap = &mmapWrapper;
  lhInfo.lhDlsym = &lhDlsym;
  lhInfo.lhMmapListFptr = &getMmappedList;
  if (syscall(SYS_arch_prctl, ARCH_GET_FS, &lhInfo.lhFsAddr) < 0) {
    DLOG(ERROR, "Could not retrieve lower half's fs. Error: %s. Exiting...\n",
         strerror(errno));
    return -1;
  }
  // FIXME: We'll just write out the lhInfo object to a file; the upper half
  // will read this file to figure out the wrapper addresses. This is ugly
  // but will work for now.
  int rc = writeLhInfoToFile();
  if (rc < 0) {
    DLOG(ERROR, "Error writing address of lhinfo to file. Exiting...\n");
    return -1;
  }
  return 0;
}
