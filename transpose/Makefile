
# Makefile for CUDA matrix transpose

# Compiler
NVCC = nvcc

# Compiler flags
CUTLASS_PATH = ../cutlass
NVCC_FLAGS = -O3 -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -arch=sm_70 -std=c++17

# Directories
BIN_DIR = bin

# Target executable
TARGET = $(BIN_DIR)/transpose

# Source file
SRC = transpose.cu

# Phony targets
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
