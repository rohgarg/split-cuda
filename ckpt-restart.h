#ifndef CKPT_RESTART_H
#define CKPT_RESTART_H

#include <signal.h>
#include <ucontext.h>

#define CKPT_SIGNAL  SIGUSR2

typedef struct __CkptRestartState
{
  ucontext_t ctx;
  void *sp;
} CkptRestartState_t;

typedef enum __CkptOrRestore
{
  CKPT,
  RESTORE,
} CkptOrRestore_t;

#endif // ifndef CKPT_RESTART_H
