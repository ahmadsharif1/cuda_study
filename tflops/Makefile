NVCC ?= nvcc
CUDA_ARCH ?= sm_90a

all: bin/tflops

bin/tflops: tflops.cu
	@mkdir -p bin
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -lcublas -o $@ $<

clean:
	rm -rf bin

.PHONY: all clean
