# Verifying Split-Process Approach for Checkpoint-Restart for CUDA

## Table of Contents

* [Idea](#idea)
* [High-level Design](#high-level-design)
  * [Start up](#start-up)
  * [Runtime logic](#runtime-logic)
  * [Checkpoint algorithm](#checkpoint-algorithm)
  * [Restart algorithm](#restart-algorithm)
* [TODO](#todo)
* [Results](#results)
* [Appendix](#appendix)
  * [Running the code](#running-the-code)
  * [Known issues](#known-issues)

## Idea

The idea here is to verify the feasibility of checkpoint-restart of CUDA
applications using the new "split-process" approach with the simplest
possible experiment, without including the complexities of DMTCP.

**The experiment:**

1. Start application with two separate "halves": an upper half
   (with the application state), and a lower half (with the CUDA
   libraries)
2. Make one simple CUDA call, which would force the CUDA libraries to
   initialize.
3. Checkpoint the upper half and kill the process.
4. Start a new process (with a new lower half).
5. Restore memory for the upper half.
6. Call the simple CUDA call one more time from the upper half.

## High-level Design

### Start up

First, we start with the lower half (the "proxy") -- a trivial executable
(with the proxy code), the Nvidia libraries, and the kernel loader code.

Next, the kernel loader (in the lower half) maps in a second heap and
a second stack region. (The second stack is going to be used by the
target application and the second ld.so, which we discuss in the next
few paragraphs.)

Then, the lower half loads in a new ld.so memory and asks the new (second)
ld.so to load the target (CUDA) application executable. The target
application and its dependencies become part of the logical upper half.

Finally, the lower half passes control to the target (CUDA) application
through the second ld.so.

By putting a trampoline around the second ld.so's mmap function, we can
control and monitor the memory regions it loads. Similarly, a trampoline
around the second ld.so's sbrk function enables us to have two separate
heaps -- one for the the upper half, and one for the lower half. Note
that the there's just one "real" heap for the process and that is what
the kernel provided to the lower half's trivial executable when it was
started.

### Runtime logic

The lower half provides a dlsym-like API to the upper half. The upper
half uses this API to figure out the addresses of the CUDA functions
(in the Nvidia libraries) in the lower half.

The target application makes CUDA calls by using the lower-half's
dlysm-like API.

### Checkpoint algorithm

The lower half provides an API that the upper half can query to figure
out what memory regions belong to the upper half. Recall that we had
inserted trampolines at start-up time on the upper half's ld.so to
monitor all the mmap calls it does.

At checkpoint time, the upper half calls this API to figure out the
regions that belong to it and saves them to a checkpoint image (along
with the current context).

### Restart algorithm

We start a new process with a new lower half. Then, we run some memory
restore code to bring back the upper half from a given checkpoint image.
Finally, we call `setcontext()` to jump back to the upper half.

(The upper half could then go back to executing the runtime logic. In
 particular, it would make some CUDA call to try to initialize the CUDA
 libraries in the upper half.)

## TODO

* [x] Add wrappers/trampolines around mmap/sbrk functions for upper-half's
       libc. In addition to lower-half's ld.so, lower-half's libc can also
       make these calls. We want to keep track of mmaps and "virtualize"
       the sbrk calls in order to force it to use the lower-half's heap.
* [x] Add a dlsym-like API in the lower half to figure out addresses of CUDA API
* [x] Test calling a CUDA function (through lower-half's dlsym API) from the
       upper half
* [x] Add checkpoint-restart logic from mini-DMTCP assignment
* [x] Test full checkpoint-restart functionality
* [x] Debug the crash in `printf()` immediately after restart in target `main()`
* [ ] Update the target addresses of trampolines in upper-half's ld.so and libc
      post restart
* [x] DLOGs in the lower half seem to crash post restart
* [ ] Debug checkpoint-restart with ASLR enabled
* [ ] Create a fake VDSO page for the upper half (See [here](#known-issues)
      for more details).

## Results

Basic checkpoint-restart works -- with some caveats. See the list of open items
in the [TODO](#todo) section.

## Appendix

### Running the code

1. Update the variables at the top of the Makefile to correspond to your system.
2. Run `make restart` to build and run the code.

### Known issues

1. Restart segfaults if ASLR is enabled; it's probably because of address
   conflicts.
2. The application might crash after restart if it tries to call
   `gettimeofday()`. Since the upper half is dynamically-linked, it
   needs the VDSO page for making calls such as, `gettimeofday()`, etc.
   And the location of the VDSO page might change post restart.
