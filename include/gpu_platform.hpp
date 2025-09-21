#pragma once

#ifdef USE_SYCL
    // SYCL/oneAPI/Intel GPU support - use specific includes to avoid deprecated headers
    #include <sycl/queue.hpp>
    #include <sycl/device.hpp>
    #include <sycl/context.hpp>
    #include <sycl/usm.hpp>
    #include <memory>

    // SYCL doesn't have direct equivalents, so we'll create wrapper types
    using gpuError_t = int;
    using gpuStream_t = void*;  // Will hold SYCL queue pointer
    using gpuEvent_t = void*;   // Will hold SYCL event pointer
    using gpuMemcpyKind = int;

    // Error codes for SYCL
    #define gpuSuccess 0
    #define gpuErrorMemoryAllocation 1
    #define gpuErrorInvalidValue 2
    #define gpuErrorInvalidDevice 3
    #define gpuErrorNoDevice 4

    // Additional CUDA error codes for compatibility
    #define cudaErrorInvalidDevice gpuErrorInvalidDevice
    #define cudaErrorNoDevice gpuErrorNoDevice

    // CUDA function compatibility defines
    #define cudaDeviceReset gpuDeviceReset

    // Memory copy kinds
    #define gpuMemcpyHostToDevice 1
    #define gpuMemcpyDeviceToHost 2
    #define gpuMemcpyDeviceToDevice 3
    #define gpuMemcpyHostToHost 4

    // Stream flags
    #define gpuStreamDefault 0
    #define gpuStreamNonBlocking 1

    // Event flags
    #define gpuEventDefault 0
    #define gpuEventDisableTiming 1

    // Host allocation flags
    #define gpuHostAllocDefault 0
    #define gpuHostAllocMapped 1
    #define gpuHostAllocWriteCombined 2

    // Device limit types
    #define gpuLimitPersistingL2CacheSize 1

    // Custom device properties structure for SYCL
    struct gpuDeviceProp {
        char name[256];
        size_t totalGlobalMem;
        int major;
        int minor;
        int multiProcessorCount;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        size_t sharedMemPerBlock;
        int warpSize;
        int clockRate;
        size_t l2CacheSize;  // Added missing field
        int maxThreadsPerMultiProcessor;  // Added missing field
    };

    // Function declarations for SYCL wrappers
    extern "C" {
        gpuError_t gpuMalloc(void** ptr, size_t size);
        gpuError_t gpuFree(void* ptr);
        gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, gpuMemcpyKind kind);
        gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count, gpuMemcpyKind kind, gpuStream_t stream);
        gpuError_t gpuMemset(void* ptr, int value, size_t count);
        gpuError_t gpuMemsetAsync(void* ptr, int value, size_t count, gpuStream_t stream);
        gpuError_t gpuMemGetInfo(size_t* free, size_t* total);
        gpuError_t gpuSetDevice(int device);
        gpuError_t gpuGetDevice(int* device);
        gpuError_t gpuGetDeviceCount(int* count);
        gpuError_t gpuGetDeviceProperties(gpuDeviceProp* prop, int device);
        gpuError_t gpuStreamCreate(gpuStream_t* stream);
        gpuError_t gpuStreamCreateWithFlags(gpuStream_t* stream, unsigned int flags);
        gpuError_t gpuStreamDestroy(gpuStream_t stream);
        gpuError_t gpuStreamSynchronize(gpuStream_t stream);
        gpuError_t gpuStreamQuery(gpuStream_t stream);
        gpuError_t gpuEventCreate(gpuEvent_t* event);
        gpuError_t gpuEventDestroy(gpuEvent_t event);
        gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream);
        gpuError_t gpuEventSynchronize(gpuEvent_t event);
        gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end);
        gpuError_t gpuEventQuery(gpuEvent_t event);
        gpuError_t gpuGetLastError(void);
        const char* gpuGetErrorString(gpuError_t error);
        gpuError_t gpuDeviceSynchronize(void);
        gpuError_t gpuHostAlloc(void** ptr, size_t size, unsigned int flags);
        gpuError_t gpuFreeHost(void* ptr);

        // Additional SYCL function declarations for missing APIs
        gpuError_t gpuDeviceReset(void);
        gpuError_t gpuDeviceSetLimit(int limit, size_t value);
        gpuError_t gpuEventCreateWithFlags(gpuEvent_t* event, unsigned int flags);
        gpuError_t gpuDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
        gpuError_t gpuStreamCreateWithPriority(gpuStream_t* stream, unsigned int flags, int priority);

        // SYCL initialization functions
        bool initialize_sycl_runtime(void);
        void cleanup_sycl_runtime(void);
        void update_base_message_sycl(const uint32_t* base_msg_words);
    }

    // Constants
    #define gpuSuccess 0
    #define gpuMemcpyHostToDevice 1
    #define gpuMemcpyDeviceToHost 2
    #define gpuMemcpyDeviceToDevice 3
    #define gpuStreamNonBlocking 1
    #define gpuStreamDefault 0
    #define gpuErrorNotReady 1

    // Device function qualifiers (not used in SYCL but kept for compatibility)
    #define __gpu_device__
    #define __gpu_global__
    #define __gpu_host__
    #define __gpu_forceinline__ inline
    #define __gpu_shared__
    #define __gpu_constant__

    // Built-in functions
    #define __gpu_popc __builtin_popcount
    #define __gpu_clz __builtin_clz

#elif USE_HIP
    // HIP/AMD GPU support
    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>

    // Type aliases for platform independence
    using gpuError_t = hipError_t;
    using gpuStream_t = hipStream_t;
    using gpuEvent_t = hipEvent_t;
    using gpuDeviceProp = hipDeviceProp_t;
    using gpuMemcpyKind = hipMemcpyKind;

    // Function aliases
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemcpyAsync hipMemcpyAsync
    #define gpuMemset hipMemset
    #define gpuMemsetAsync hipMemsetAsync
    #define gpuMemGetInfo hipMemGetInfo
    #define gpuSetDevice hipSetDevice
    #define gpuGetDevice hipGetDevice
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuGetDeviceProperties hipGetDeviceProperties
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamCreateWithFlags hipStreamCreateWithFlags
    #define gpuStreamCreateWithPriority hipStreamCreateWithPriority
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    #define gpuStreamQuery hipStreamQuery
    #define gpuEventCreate hipEventCreate
    #define gpuEventCreateWithFlags hipEventCreateWithFlags
    #define gpuEventDestroy hipEventDestroy
    #define gpuEventRecord hipEventRecord
    #define gpuEventSynchronize hipEventSynchronize
    #define gpuEventElapsedTime hipEventElapsedTime
    #define gpuEventQuery hipEventQuery  // ADD THIS LINE
    #define gpuGetLastError hipGetLastError
    #define gpuGetErrorString hipGetErrorString
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuHostAlloc hipHostMalloc
    #define gpuFreeHost hipHostFree
    #define gpuDeviceSetLimit hipDeviceSetLimit
    #define gpuDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange

    // Constants
    #define gpuSuccess hipSuccess
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define gpuStreamNonBlocking hipStreamNonBlocking
    #define gpuStreamDefault hipStreamDefault
    #define gpuEventDisableTiming hipEventDisableTiming
    #define gpuHostAllocMapped hipHostMallocMapped
    #define gpuHostAllocWriteCombined hipHostMallocWriteCombined
    #define gpuLimitPersistingL2CacheSize hipLimitPersistingL2CacheSize
    #define gpuErrorNotReady hipErrorNotReady  // ADD THIS LINE

    // Device function qualifiers
    #define __gpu_device__ __device__
    #define __gpu_global__ __global__
    #define __gpu_host__ __host__
    #define __gpu_forceinline__ __forceinline__
    #define __gpu_shared__ __shared__
    #define __gpu_constant__ __constant__

    // Built-in functions
    #define __gpu_popc __popc
    #define __gpu_clz __clz

#else
    // CUDA/NVIDIA GPU support
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>

    // Type aliases for platform independence
    using gpuError_t = cudaError_t;
    using gpuStream_t = cudaStream_t;
    using gpuEvent_t = cudaEvent_t;
    using gpuDeviceProp = cudaDeviceProp;
    using gpuMemcpyKind = cudaMemcpyKind;

    // Function aliases
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyAsync cudaMemcpyAsync
    #define gpuMemset cudaMemset
    #define gpuMemsetAsync cudaMemsetAsync
    #define gpuMemGetInfo cudaMemGetInfo
    #define gpuSetDevice cudaSetDevice
    #define gpuGetDevice cudaGetDevice
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuGetDeviceProperties cudaGetDeviceProperties
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
    #define gpuStreamCreateWithPriority cudaStreamCreateWithPriority
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    #define gpuStreamQuery cudaStreamQuery
    #define gpuEventCreate cudaEventCreate
    #define gpuEventCreateWithFlags cudaEventCreateWithFlags
    #define gpuEventDestroy cudaEventDestroy
    #define gpuEventRecord cudaEventRecord
    #define gpuEventSynchronize cudaEventSynchronize
    #define gpuEventElapsedTime cudaEventElapsedTime
    #define gpuEventQuery cudaEventQuery  // ADD THIS LINE
    #define gpuGetLastError cudaGetLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuHostAlloc cudaHostAlloc
    #define gpuFreeHost cudaFreeHost
    #define gpuDeviceSetLimit cudaDeviceSetLimit
    #define gpuDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange

    // Constants
    #define gpuSuccess cudaSuccess
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuStreamNonBlocking cudaStreamNonBlocking
    #define gpuStreamDefault cudaStreamDefault
    #define gpuEventDisableTiming cudaEventDisableTiming
    #define gpuHostAllocMapped cudaHostAllocMapped
    #define gpuHostAllocWriteCombined cudaHostAllocWriteCombined
    #define gpuLimitPersistingL2CacheSize cudaLimitPersistingL2CacheSize
    #define gpuErrorNotReady cudaErrorNotReady  // ADD THIS LINE

    // Device function qualifiers
    #define __gpu_device__ __device__
    #define __gpu_global__ __global__
    #define __gpu_host__ __host__
    #define __gpu_forceinline__ __forceinline__
    #define __gpu_shared__ __shared__
    #define __gpu_constant__ __constant__

    // Built-in functions
    #define __gpu_popc __popc
    #define __gpu_clz __clz

#endif

// Common GPU error checking macro
#define GPU_CHECK(call) \
    do { \
        gpuError_t error = call; \
        if (error != gpuSuccess) { \
            fprintf(stderr, "GPU Error: %s at %s:%d\n", \
                    gpuGetErrorString(error), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// Macro to ignore return value warnings
#define GPU_IGNORE_RESULT(call) \
    do { \
        gpuError_t _ignored_result = call; \
        (void)_ignored_result; \
    } while(0)

// Platform-specific includes for device code
#if defined(__CUDACC__) || defined(__HIPCC__)
    // Device code is being compiled
    #define GPU_DEVICE_CODE
#endif