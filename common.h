#ifndef COMMON_H
#define COMMON_H

#include <link.h>
#include <string.h>

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
} LowerHalfInfo_t;

extern LowerHalfInfo_t lhInfo;

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

#endif // ifndef COMMON_H
