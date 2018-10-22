#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

// Returns pointer to argc, given a pointer to end of stack
static inline void*
GET_ARGC_ADDR(const void* stackEnd)
{
  return (void*)((uintptr_t)(stackEnd) + sizeof(uintptr_t));
}

// Returns pointer to argv[0], given a pointer to end of stack
static inline void*
GET_ARGV_ADDR(const void* stackEnd)
{
  return (void*)((unsigned long)(stackEnd) + 2 * sizeof(uintptr_t));
}

// Returns pointer to env[0], given a pointer to end of stack
static inline void*
GET_ENV_ADDR(char **argv, int argc)
{
  return (void*)&argv[argc + 1];
}

// Returns a pointer to aux vector, given a pointer to the environ vector
// on the stack
static inline ElfW(auxv_t)*
GET_AUXV_ADDR(const char **env)
{
  ElfW(auxv_t) *auxvec;
  const char **evp = env;
  while (*evp++ != NULL);
  auxvec = (ElfW(auxv_t) *) evp;
  return auxvec;
}

void runRtld();
void* sbrkWrapper(intptr_t );
void* mmapWrapper(void *, size_t , int , int , int , off_t );
void* getEndofHeap();
void setEndOfHeap(void *);

#endif // ifndef KERNEL_LOADER_H
