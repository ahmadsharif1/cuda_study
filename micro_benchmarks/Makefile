NVCC ?= nvcc
CUDA_ARCH ?= native

all: bin/copy_bw bin/l2_bw

bin/copy_bw: copy_bw.cu
	@mkdir -p bin
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -o $@ $<

bin/l2_bw: l2_bw.cu
	@mkdir -p bin
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -o $@ $<

clean:
	rm -rf bin

.PHONY: all clean
