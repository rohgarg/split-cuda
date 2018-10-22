FILE=kernel-loader

RTLD_PATH=/lib64/ld-2.27.so

KERNEL_LOADER_OBJS=${FILE}.o procmapsutils.o custom-loader.o mmap-wrapper.o sbrk-wrapper.o cuda-lh-if.o
TARGET_OBJS=target.o
TARGET_PRELOAD_LIB_OBJS=upper-half-wrappers.o

CUDA_INCLUDE_PATH=/usr/local/cuda/include/

CFLAGS=-g3 -O0 -fPIC -I. -I${CUDA_INCLUDE_PATH} -c -std=gnu11
KERNEL_LOADER_CFLAGS=-DSTANDALONE

KERNEL_LOADER_BIN=kernel-loader.exe
TARGET_BIN=t.exe
TARGET_PRELOAD_LIB=libuhwrappers.so

run: ${KERNEL_LOADER_BIN} ${TARGET_BIN} ${TARGET_PRELOAD_LIB}
	UH_PRELOAD=$$PWD/${TARGET_PRELOAD_LIB} TARGET_LD=${RTLD_PATH} ./$< $$PWD/${TARGET_BIN} arg1 arg2 arg3

gdb: ${KERNEL_LOADER_BIN} ${TARGET_BIN} ${TARGET_PRELOAD_LIB}
	UH_PRELOAD=$$PWD/${TARGET_PRELOAD_LIB} TARGET_LD=${RTLD_PATH} gdb --args ./$< $$PWD/${TARGET_BIN} arg1 arg2 arg3

.c.o:
	gcc ${CFLAGS} $< -o $@

${FILE}.o: ${FILE}.c
	gcc ${CFLAGS} ${KERNEL_LOADER_CFLAGS} $< -o $@

${TARGET_BIN}: ${TARGET_OBJS}
	gcc $< -o $@

${TARGET_PRELOAD_LIB}: ${TARGET_PRELOAD_LIB_OBJS}
	gcc -shared $< -o $@

# Apparently, Nvidia libraries don't like -pie; so, we are forced
# to link the kernel loader (which is really just emulating the lower
# half) to a fixed address (0x800000)
${KERNEL_LOADER_BIN}: ${KERNEL_LOADER_OBJS}
	nvcc -Xlinker -Ttext-segment -Xlinker 0x800000 --cudart shared $^ -o $@ -lcuda

vi vim:
	vim ${FILE}.c

tags:
	gtags .

dist: clean
	(dir=`basename $$PWD` && cd .. && tar zcvf $$dir.tgz $$dir)
	(dir=`basename $$PWD` && ls -l ../$$dir.tgz)

clean:
	rm -f ${KERNEL_LOADER_OBJS} ${TARGET_OBJS} ${KERNEL_LOADER_BIN} ${TARGET_BIN} ${TARGET_PRELOAD_LIB_OBJS} ${TARGET_PRELOAD_LIB} GTAGS GRTAGS GPATH

.PHONY: dist vi vim clean gdb tags
