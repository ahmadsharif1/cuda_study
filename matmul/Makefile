
# Makefile for CUDA matrix multiplication

# Compiler
NVCC = nvcc

# GPU architecture: override via command line (make CUDA_ARCH=sm_80) or environment variable
# Defaults to sm90a (change to native for your GPU)
CUDA_ARCH ?= sm_90a

# Compiler flags
CUTLASS_PATH = ../cutlass
NVCC_FLAGS = -lineinfo -g -O3 -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -arch=$(CUDA_ARCH) -std=c++17 -lcublas

# Directories
BIN_DIR = bin

# Target executable
TARGET = $(BIN_DIR)/matmul

# Source file
SRC = matmul.cu

# Phony targets
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
