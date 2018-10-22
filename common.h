#ifndef COMMON_H
#define COMMON_H

#include <link.h>
#include <stdio.h>
#include <string.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

// Logging levels
#define NOISE 3 // Noise!
#define INFO  2 // Informational logs
#define ERROR 1 // Highest error/exception level

#ifndef DEBUG_LEVEL
// Let's announce errors out loud
# define DEBUG_LEVEL 1
#endif // ifndef DEBUG_LEVEL

#define VA_ARGS(...)  , ##__VA_ARGS__
#define DLOG(LOG_LEVEL, fmt, ...)                                              \
do {                                                                           \
  if (DEBUG_LEVEL) {                                                           \
    if (LOG_LEVEL <= DEBUG_LEVEL)                                              \
      fprintf(stderr, "[%s +%d]: " fmt, __FILE__,                              \
              __LINE__ VA_ARGS(__VA_ARGS__));                                  \
  }                                                                            \
} while(0)

// Based on the entries in /proc/<pid>/stat as described in `man 5 proc`
enum Procstat_t
{
   PID = 1,
   COMM,   // 2
   STATE,  // 3
   PPID,   // 4
   NUM_THREADS = 19,
   STARTSTACK = 27,
};

#define PAGE_SIZE 0x1000ULL

// FIXME: 0x1000 is one page; Use sysconf(PAGESIZE) instead.
#define ROUND_DOWN(x) ((unsigned long long)(x) \
                      & ~(unsigned long long)(PAGE_SIZE - 1))
#define ROUND_UP(x)  (((unsigned long long)(x) + PAGE_SIZE - 1) & \
                      ~(PAGE_SIZE - 1))
#define PAGE_OFFSET(x)  ((x) & (PAGE_SIZE - 1))

// TODO: This is very x86-64 specific; support other architectures??
#define eax rax
#define ebx rbx
#define ecx rcx
#define edx rax
#define ebp rbp
#define esi rsi
#define edi rdi
#define esp rsp
#define CLEAN_FOR_64_BIT_HELPER(args ...) # args
#define CLEAN_FOR_64_BIT(args ...)        CLEAN_FOR_64_BIT_HELPER(args)

typedef struct __LowerHalfInfo
{
  void *lhSbrk;
  void *lhMmap;
  void *lhDlsym;
  unsigned long lhFsAddr;
  void *lhMmapListFptr;
} LowerHalfInfo_t;

extern LowerHalfInfo_t lhInfo;

// Helper macro to be used whenever jumping into the lower half from the
// upper half.
#define JUMP_TO_LOWER_HALF(lhFs)                                               \
  do {                                                                         \
  unsigned long upperHalfFs;                                                   \
  syscall(SYS_arch_prctl, ARCH_GET_FS, &upperHalfFs);                          \
  syscall(SYS_arch_prctl, ARCH_SET_FS, lhFs);

// Helper macro to be used whenever making a returning from the lower half to
// the upper half.
#define RETURN_TO_UPPER_HALF()                                                 \
  syscall(SYS_arch_prctl, ARCH_SET_FS, &upperHalfFs);                          \
  } while (0)

#define LH_FILE_NAME "./addr.bin"

#define FOREACH_FNC(MACRO) \
    MACRO(cuInit)

#define GENERATE_ENUM(ENUM) Cuda_Fnc_##ENUM,
#define GENERATE_FNC_PTR(FNC) &FNC,

typedef enum __Cuda_Fncs {
  Cuda_Fnc_NULL,
  FOREACH_FNC(GENERATE_ENUM)
  Cuda_Fnc_Invalid,
} Cuda_Fncs_t;

void* lhDlsym(Cuda_Fncs_t type);
typedef void* (*LhDlsym_t)(Cuda_Fncs_t type);


void** getMmappedList(int *num);
typedef void** (*GetMmappedListFptr_t)(int *num);

#endif // ifndef COMMON_H
