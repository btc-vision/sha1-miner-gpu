#ifndef GPU_PLATFORM_HPP
#define GPU_PLATFORM_HPP

// Platform detection and abstraction layer
#ifdef USE_HIP
    #include <hip/hip_runtime.h>
    #include <hip/hip_cooperative_groups.h>

    // Map CUDA types to HIP types
    #define gpuError_t hipError_t
    #define gpuSuccess hipSuccess
    #define gpuGetErrorString hipGetErrorString
    #define gpuGetLastError hipGetLastError
    #define gpuDeviceProp hipDeviceProp_t
    #define gpuGetDeviceProperties hipGetDeviceProperties
    #define gpuSetDevice hipSetDevice
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemcpyAsync hipMemcpyAsync
    #define gpuMemset hipMemset
    #define gpuMemsetAsync hipMemsetAsync
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuHostAlloc hipHostMalloc
    #define gpuHostAllocMapped hipHostMallocMapped
    #define gpuHostAllocWriteCombined hipHostMallocWriteCombined
    #define gpuFreeHost hipHostFree
    #define gpuStream_t hipStream_t
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamCreateWithFlags hipStreamCreateWithFlags
    #define gpuStreamCreateWithPriority hipStreamCreateWithPriority
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    #define gpuStreamQuery hipStreamQuery
    #define gpuStreamNonBlocking hipStreamNonBlocking
    #define gpuEvent_t hipEvent_t
    #define gpuEventCreate hipEventCreate
    #define gpuEventCreateWithFlags hipEventCreateWithFlags
    #define gpuEventDestroy hipEventDestroy
    #define gpuEventDisableTiming hipEventDisableTiming
    #define gpuDeviceSetLimit hipDeviceSetLimit
    #define gpuLimitPersistingL2CacheSize hipLimitPersistingL2CacheSize
    #define gpuMemGetInfo hipMemGetInfo
    #define gpuDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuEventRecord hipEventRecord
    #define gpuEventSynchronize hipEventSynchronize
    #define gpuEventElapsedTime hipEventElapsedTime
    #define gpuDeviceSetLimit hipDeviceSetLimit

    #define GPU_LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
        hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)

    #define __gpu_device__ __device__
    #define __gpu_global__ __global__
    #define __gpu_host__ __host__
    #define __gpu_constant__ __constant__
    #define __gpu_shared__ __shared__
    #define __gpu_forceinline__ __forceinline__

    // HIP-specific intrinsics mapping
    #define __gpu_clz(x) __clz(x)
    #define __gpu_popc(x) __popc(x)
    #define __gpu_funnelshift_l(x, y, n) __funnelshift_l(x, y, n)
    #define __gpu_byte_perm(x, y, s) amd_byteperm(x, y, s)
    #define __gpu_syncthreads() __syncthreads()
    #define __gpu_threadfence() __threadfence()

    // AMD-specific wavefront size
    #define GPU_WARP_SIZE 64
    #define GPU_WAVEFRONT_SIZE 64

#else
// Default to CUDA
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuHostAlloc cudaHostAlloc
#define gpuHostAllocMapped cudaHostAllocMapped
#define gpuHostAllocWriteCombined cudaHostAllocWriteCombined
#define gpuFreeHost cudaFreeHost
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamCreateWithPriority cudaStreamCreateWithPriority
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamQuery cudaStreamQuery
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDestroy cudaEventDestroy
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuDeviceSetLimit cudaDeviceSetLimit
#define gpuLimitPersistingL2CacheSize cudaLimitPersistingL2CacheSize
#define gpuMemGetInfo cudaMemGetInfo
#define gpuDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceSetLimit cudaDeviceSetLimit

#define GPU_LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
        kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)

#define __gpu_device__ __device__
#define __gpu_global__ __global__
#define __gpu_host__ __host__
#define __gpu_constant__ __constant__
#define __gpu_shared__ __shared__
#define __gpu_forceinline__ __forceinline__

// CUDA intrinsics
#define __gpu_clz(x) __clz(x)
#define __gpu_popc(x) __popc(x)
#define __gpu_funnelshift_l(x, y, n) __funnelshift_l(x, y, n)
#define __gpu_byte_perm(x, y, s) __byte_perm(x, y, s)
#define __gpu_syncthreads() __syncthreads()
#define __gpu_threadfence() __threadfence()

// NVIDIA warp size
#define GPU_WARP_SIZE 32
#define GPU_WAVEFRONT_SIZE 32
#endif

// Platform-independent GPU architecture detection
inline const char *getGPUPlatformName() {
#ifdef USE_HIP
    return "HIP/ROCm";
#else
    return "CUDA";
#endif
}

// Helper function to check if we're on AMD GPU
inline bool isAMDGPU() {
#ifdef USE_HIP
    return true;
#else
    return false;
#endif
}

// Get optimal thread block size based on platform
inline int getOptimalThreadsPerBlock(int multiProcessorCount) {
#ifdef USE_HIP
    // AMD GPUs typically prefer 256 threads per block
    return 256;
#else
    // NVIDIA GPUs can vary, but 256 is a good default
    return 256;
#endif
}

// Get optimal blocks per SM/CU based on platform
inline int getOptimalBlocksPerSM(int major, int minor) {
#ifdef USE_HIP
    // AMD RDNA architecture
    if (major >= 10) {
        return 8;  // RDNA/RDNA2/RDNA3
    }
    // AMD GCN architecture
    return 4;
#else
    // NVIDIA architectures
    if (major >= 8) {
        return 16; // Ampere and newer
    } else if (major == 7) {
        return 8; // Volta/Turing
    } else if (major == 6) {
        return 8; // Pascal
    }
    return 4; // Older architectures
#endif
}

// Platform-specific memory alignment
inline size_t getMemoryAlignment() {
#ifdef USE_HIP
    return 256;  // AMD prefers 256-byte alignment
#else
    return 128; // NVIDIA typically uses 128-byte alignment
#endif
}

// Helper for AMD-specific byteperm implementation
#ifdef USE_HIP
__device__ inline uint32_t amd_byteperm(uint32_t x, uint32_t y, uint32_t s) {
    // AMD doesn't have byte_perm, emulate with bit operations
    union { uint32_t i; uint8_t b[4]; } xu, yu, ru;
    xu.i = x;
    yu.i = y;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t sel = (s >> (i * 8)) & 0xF;
        if (sel < 4) {
            ru.b[i] = xu.b[sel];
        } else {
            ru.b[i] = yu.b[sel - 4];
        }
    }
    return ru.i;
}
#endif

#endif // GPU_PLATFORM_HPP