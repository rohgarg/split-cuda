#ifndef UTILS_H
#define UTILS_H

#include <linux/limits.h>

ssize_t writeAll(int , const void *, size_t );
ssize_t readAll(int , void* , size_t );
int checkLibrary(int, const char*, char *, size_t );

#endif // ifndef UTILS_H
