
# Makefile for CUDA matrix transpose

# Compiler
NVCC = nvcc

# Compiler flags
CUTLASS_PATH = /home/ahmad/cuda/cutlass
NVCC_FLAGS = -O3 -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -arch=sm_70 -std=c++17

# Target executable
TARGET = transpose

# Source file
SRC = transpose.cu

# Phony targets
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
