include tensorflow/lite/micro/tools/make/path.inc

export PATH := $(INTEL_FPGA_PATH)/18.1/nios2eds/bin/gnu/H-x86_64-pc-linux-gnu/bin:$(PATH)
TARGET_TOOLCHAIN_PREFIX := nios2-elf-
TARGET_TOOLCHAIN_ROOT := $(INTEL_FPGA_PATH)/18.1/nios2eds/bin/gnu/H-x86_64-pc-linux-gnu/bin/

BSP_ROOT_DIR := $(RTES_PATH)/miniproject/sw/cpu_0/miniproject_cpu_0_bsp
BSP_INCLUDE_FILE := $(BSP_ROOT_DIR)/public.mk
ALT_LIBRARY_ROOT_DIR := $(BSP_ROOT_DIR)
include $(BSP_INCLUDE_FILE)

INCLUDES += $(addprefix -I, $(ALT_INCLUDE_DIRS)) 
INCLUDES += $(addprefix -I, $(ALT_INCLUDE_DIRS_TO_APPEND))

CCFLAGS += $(ALT_CFLAGS)
CXXFLAGS += $(ALT_CFLAGS)