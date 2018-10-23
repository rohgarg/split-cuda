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
  RUNNING,
  POST_RESUME,
  POST_RESTART,
} CkptOrRestore_t;

// Public API:
//   Returns POST_RESUME, if resuming from a checkpoint,
//   Returns POST_RESTORE, if restarting from a checkpoint
CkptOrRestore_t doCheckpoint() __attribute__((weak));
#define doCheckpoint() (doCheckpoint ? doCheckpoint() : 0)

void restoreCheckpoint(const char *);

#endif // ifndef CKPT_RESTART_H
