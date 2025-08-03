include make.conf

SRC_DIR := ../src
SRC_MSH_DIR := ./mesh
VPATH := $(SRC_DIR) $(SRC_MSH_DIR)

# Host configuration
HOS_CC := mpic++  #Host端使用的 编译器
HTHREADS_INCLUDE_PATH:=${HTHREADS_ROOT_PATH}/include 
HTHREADS_LIB_PATH:=${HTHREADS_ROOT_PATH}/lib
HOS_LDFLAGS := -L${HTHREADS_LIB_PATH} -lhthread_host -lm -lpthread  #链接的库
#主机端编译头文件
HOS_CFLAGS := -O3 -std=c++17 -Wall -I${HTHREADS_INCLUDE_PATH} -I$(SRC_DIR) -I$(SRC_MSH_DIR) #-I/vol8/home/hnu_test04/zxc/include 

# Device compiler configuration
DEV_CC_BIN_PATH := ${DEV_CC_ROOT_PATH}/bin
DEV_CC := ${DEV_CC_BIN_PATH}/MT-3000-gcc
DEV_LD := ${DEV_CC_BIN_PATH}/MT-3000-ld
DEV_MAKEDAT := ${DEV_CC_BIN_PATH}/MT-3000-makedat
DEV_CC_INCLUDE_PATH := ${DEV_CC_ROOT_PATH}/include
DEV_CC_LIB_PATH := ${DEV_CC_ROOT_PATH}/lib

DEV_INCLUDE_PATH := ${DEV_CC_INCLUDE_PATH} ${HTHREADS_INCLUDE_PATH}
DEV_INCLUDE_FLAGS := $(foreach i, ${DEV_INCLUDE_PATH}, -I${i})
DEV_WALLFLAGS := -Wall -Wno-attributes -Wno-unused-function
DEV_CFLAGS := $(DEV_WALLFLAGS) -O2 -fenable-m3000 -ffunction-sections \
              -flax-vector-conversions ${DEV_INCLUDE_FLAGS}

DEV_LIBRARY_PATH := ${DEV_CC_LIB_PATH} ${HTHREADS_LIB_PATH}
DEV_LDFLAGS := $(foreach i, ${DEV_LIBRARY_PATH}, -L${i})
DEV_LDFLAGS += --gc-sections -T$(HTHREADS_LIB_PATH)/dsp.lds -lhthread_device \
               ${DEV_CC_LIB_PATH}/slib3000.a

# MPI configuration
MPI_RUN := /vol8/home/hnu_test04/zxc/bin/mpirun


###############################################################################
.PHONY: ALL
ALL: mesh

# Device build rules
%.dev.o: %.dev.c
	$(DEV_CC) -c $(DEV_CFLAGS) $^ -o $@

%.dev.out: %.dev.o
	$(DEV_LD) $^ $(DEV_LDFLAGS) -o $@

%.dev.dat: %.dev.out
	$(DEV_MAKEDAT) -J $^

# Host build rule with MPI support but using gcc
mesh.hos: mesh.hos.c
	$(HOS_CC) -o $@ $^ $(HOS_CFLAGS) $(HOS_LDFLAGS)

# Combined target
mesh: mesh.hos mesh.dev.dat 

# Run rule
.PHONY: run
run: mesh
	$(MPI_RUN) -np 16 ./mesh.hos

# Clean rule
.PHONY: clean
clean:
	rm -rf *.out *.dat *.asm *.o
	rm -rf mesh.hos
.PHONY : all clean
