#ifndef CKPT_RESTART_H
#define CKPT_RESTART_H

#include <signal.h>
#include <ucontext.h>

#define CKPT_SIGNAL  SIGUSR2

typedef struct __CkptRestartState
{
  ucontext_t ctx;
  void *sp;
  void *fsAddr; // fs is perhaps not saved as part of getcontext
} CkptRestartState_t;

typedef enum __CkptOrRestore
{
  CKPT,
  RESTORE,
} CkptOrRestore_t;

void restoreCheckpoint(const char *);

#endif // ifndef CKPT_RESTART_H
