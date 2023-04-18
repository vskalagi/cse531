NVFLAGS  := -arch=sm_86 -O2 -std=c++11
CXXFLAGS := -std=c++11
all: apsp

apsp: apsp.cu
	/usr/local/cuda/bin/nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" -o $@ $?

clean:
	rm -f apsp