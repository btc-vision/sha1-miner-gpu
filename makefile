# SHA-1 Near-Collision Miner Makefile

# Compiler settings
NVCC = nvcc
CXX = g++
CUDA_PATH ?= /usr/local/cuda

# Target GPU architectures
# Adjust based on your GPU:
# - sm_75: Turing (RTX 20xx)
# - sm_80: Ampere (RTX 30xx, A100)
# - sm_86: Ampere (RTX 30xx consumer)
# - sm_89: Ada Lovelace (RTX 40xx)
# - sm_90: Hopper (H100)
# - sm_120: Blackwell (RTX 50xx, H200)
GPU_ARCH = -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_90,code=sm_90 \
           -gencode arch=compute_120,code=sm_120

# CUDA compilation flags - optimized for mining workload
NVCC_FLAGS = -O3 --use_fast_math -std=c++17 $(GPU_ARCH)
NVCC_FLAGS += -Xcompiler -O3,-march=native,-fopenmp,-ffast-math
NVCC_FLAGS += -Xptxas -O3,-v,-dlcm=ca,-dscm=wt
NVCC_FLAGS += --extra-device-vectorization --restrict
NVCC_FLAGS += -maxrregcount=72 --optimize 3

# C++ compilation flags
CXX_FLAGS = -O3 -std=c++17 -march=native -mtune=native -fopenmp
CXX_FLAGS += -ffast-math -funroll-loops -finline-functions
CXX_FLAGS += -Wall -Wextra

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -G -g -lineinfo
# CXX_FLAGS += -g -O0

# Include and library paths
INCLUDES = -I$(CUDA_PATH)/include -I.
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lpthread

# Source files
CXX_SOURCES = main.cpp sha1_host.cpp cxxsha1.cpp
CUDA_SOURCES = sha1_kernel.cu
TEST_SOURCES = verify_sha1.cpp cxxsha1.cpp

# Object files
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.test.o)

# Targets
TARGET = sha1_miner
TEST_TARGET = verify_sha1

# Default target
all: $(TARGET) $(TEST_TARGET)

# Main executable
$(TARGET): $(CXX_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# Test executable
$(TEST_TARGET): $(TEST_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ -lpthread

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ test files
%.test.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Benchmark
benchmark: $(TARGET)
	./$(TARGET) --benchmark

# Quick benchmark (1 minute per difficulty)
quick-bench: $(TARGET)
	./$(TARGET) --gpu 0 --difficulty 80 --duration 60
	./$(TARGET) --gpu 0 --difficulty 100 --duration 60
	./$(TARGET) --gpu 0 --difficulty 120 --duration 60

# Profile with nsys
profile: $(TARGET)
	nsys profile --stats=true --output=sha1_profile ./$(TARGET) --benchmark

# Profile kernel with ncu
profile-kernel: $(TARGET)
	ncu --kernel-name sha1_near_collision_kernel --launch-skip 100 --launch-count 10 \
	    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
	    ./$(TARGET) --gpu 0 --difficulty 100 --duration 30

# Clean build files
clean:
	rm -f $(TARGET) $(TEST_TARGET) *.o *.test.o

# Clean everything including profiles
distclean: clean
	rm -f *.nsys-rep *.ncu-rep *.qdrep

# Install (adjust PREFIX as needed)
PREFIX ?= /usr/local
install: $(TARGET)
	install -D -m 755 $(TARGET) $(PREFIX)/bin/$(TARGET)

# Uninstall
uninstall:
	rm -f $(PREFIX)/bin/$(TARGET)

# Help
help:
	@echo "SHA-1 Near-Collision Miner - Makefile targets:"
	@echo ""
	@echo "  make              - Build the miner and test program"
	@echo "  make test         - Run verification tests"
	@echo "  make benchmark    - Run full benchmark"
	@echo "  make quick-bench  - Run quick benchmark (3 difficulties)"
	@echo "  make profile      - Profile with Nsight Systems"
	@echo "  make profile-kernel - Profile kernel with Nsight Compute"
	@echo "  make clean        - Remove build files"
	@echo "  make distclean    - Remove all generated files"
	@echo "  make install      - Install to system"
	@echo ""
	@echo "GPU Architecture Selection:"
	@echo "  Edit GPU_ARCH in Makefile to match your GPU"
	@echo ""
	@echo "Debug Build:"
	@echo "  Uncomment debug flags in Makefile"

.PHONY: all clean distclean test benchmark quick-bench profile profile-kernel install uninstall help