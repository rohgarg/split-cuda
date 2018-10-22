#ifndef CKPT_RESTART_H
#define CKPT_RESTART_H

#include <ucontext.h>

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
