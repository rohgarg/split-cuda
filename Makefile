# NOTE: Update the following variables for your system
CC=gcc
LD=gcc
NVCC=/usr/local/cuda/bin/nvcc
RTLD_PATH=/lib64/ld-2.27.so
CUDA_INCLUDE_PATH=/usr/local/cuda/include/

FILE=kernel-loader
KERNEL_LOADER_OBJS=${FILE}.o procmapsutils.o custom-loader.o mmap-wrapper.o sbrk-wrapper.o cuda-lh-if.o mem-restore.o utils.o trampoline_setup.o
TARGET_OBJS=target.o
TARGET_PRELOAD_LIB_OBJS=upper-half-wrappers.o upper-half-cuda-wrappers.o mem-ckpt.o procmapsutils.o utils.o trampoline_setup.o

NVCC_FLAGS=-Xlinker -Ttext-segment -Xlinker 0x800000 --cudart shared 
CFLAGS=-g3 -O0 -fPIC -I. -I${CUDA_INCLUDE_PATH} -c -std=gnu11 -Wall -Werror
KERNEL_LOADER_CFLAGS=-DSTANDALONE

KERNEL_LOADER_BIN=kernel-loader.exe
TARGET_BIN=t.exe
TARGET_PRELOAD_LIB=libuhwrappers.so

ckpt.img: ${KERNEL_LOADER_BIN} ${TARGET_BIN} ${TARGET_PRELOAD_LIB} disableASLR
	UH_PRELOAD=$$PWD/${TARGET_PRELOAD_LIB} TARGET_LD=${RTLD_PATH} ./$< $$PWD/${TARGET_BIN} arg1 arg2 arg3

run: ${KERNEL_LOADER_BIN} ${TARGET_BIN} ${TARGET_PRELOAD_LIB} disableASLR
	UH_PRELOAD=$$PWD/${TARGET_PRELOAD_LIB} TARGET_LD=${RTLD_PATH} ./$< $$PWD/${TARGET_BIN} arg1 arg2 arg3

gdb: ${KERNEL_LOADER_BIN} ${TARGET_BIN} ${TARGET_PRELOAD_LIB}
	UH_PRELOAD=$$PWD/${TARGET_PRELOAD_LIB} TARGET_LD=${RTLD_PATH} gdb --args ./$< $$PWD/${TARGET_BIN} arg1 arg2 arg3

restart: ${KERNEL_LOADER_BIN} ckpt.img disableASLR
	UH_PRELOAD=$$PWD/${TARGET_PRELOAD_LIB} TARGET_LD=${RTLD_PATH} ./$< --restore ./ckpt.img

disableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` == 0 ] || sudo sh -c 'echo 0 > /proc/sys/kernel/randomize_va_space'

enableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` -ne 2 ] && sudo sh -c 'echo 2 > /proc/sys/kernel/randomize_va_space'

.c.o:
	${CC} ${CFLAGS} $< -o $@

${FILE}.o: ${FILE}.c
	${CC} ${CFLAGS} ${KERNEL_LOADER_CFLAGS} $< -o $@

${TARGET_BIN}: ${TARGET_OBJS}
	${LD} $< -o $@

${TARGET_PRELOAD_LIB}: ${TARGET_PRELOAD_LIB_OBJS}
	${LD} -shared $^ -o $@

# Apparently, Nvidia libraries don't like -pie; so, we are forced
# to link the kernel loader (which is really just emulating the lower
# half) to a fixed address (0x800000)
${KERNEL_LOADER_BIN}: ${KERNEL_LOADER_OBJS}
	${NVCC} ${NVCC_FLAGS} $^ -o $@ -lcuda -ldl

vi vim:
	vim ${FILE}.c

tags:
	gtags .

dist: clean
	(dir=`basename $$PWD` && cd .. && tar zcvf $$dir.tgz $$dir)
	(dir=`basename $$PWD` && ls -l ../$$dir.tgz)

tidy:
	rm -f ./ckpt.img ./addr.bin

clean: tidy
	rm -f ${KERNEL_LOADER_OBJS} ${TARGET_OBJS} ${KERNEL_LOADER_BIN} \
	      ${TARGET_BIN} ${TARGET_PRELOAD_LIB_OBJS} ${TARGET_PRELOAD_LIB} \
		  GTAGS GRTAGS GPATH

.PHONY: dist vi vim clean gdb tags tidy restart run enableASLR disableASLR
