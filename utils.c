#include <errno.h>
#include <unistd.h>

#include "utils.h"

ssize_t
writeAll(int fd, const void *buf, size_t count)
{
  const char *ptr = (const char *)buf;
  size_t num_written = 0;

  do {
    ssize_t rc = write(fd, ptr + num_written, count - num_written);
    if (rc == -1) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      } else {
        return rc;
      }
    } else if (rc == 0) {
      break;
    } else { // else rc > 0
      num_written += rc;
    }
  } while (num_written < count);
  return num_written;
}

ssize_t
readAll(int fd, void *buf, size_t count)
{
  char *ptr = (char *)buf;
  size_t num_read = 0;

  for (num_read = 0; num_read < count;) {
    ssize_t rc = read(fd, ptr + num_read, count - num_read);
    if (rc == -1) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      } else {
        return -1;
      }
    } else if (rc == 0) {
      break;
    } else { // else rc > 0
      num_read += rc;
    }
  }
  return num_read;
}
