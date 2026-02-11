
# Makefile for CUDA matrix multiplication

# Compiler
NVCC = nvcc

# GPU architecture: override via command line (make CUDA_ARCH=sm_80) or environment variable
# Defaults to native (auto-detect host GPU)
CUDA_ARCH ?= native

# Compiler flags
NVCC_FLAGS = -g -O3 -arch=$(CUDA_ARCH)

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
