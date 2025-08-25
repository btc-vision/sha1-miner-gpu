#pragma once

#ifdef USE_HIP
    // HIP/AMD GPU support
    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>

// Type aliases for platform independence
using gpuError_t    = hipError_t;
using gpuStream_t   = hipStream_t;
using gpuEvent_t    = hipEvent_t;
using gpuDeviceProp = hipDeviceProp_t;
using gpuMemcpyKind = hipMemcpyKind;

    // Function aliases
    #define gpuHostAllocDefault             0
    #define gpuHostAllocPortable            hipHostMallocPortable
    #define gpuHostAllocCoherent            hipHostMallocCoherent
    #define gpuMalloc                       hipMalloc
    #define gpuFree                         hipFree
    #define gpuMemcpy                       hipMemcpy
    #define gpuMemcpyAsync                  hipMemcpyAsync
    #define gpuMemset                       hipMemset
    #define gpuMemsetAsync                  hipMemsetAsync
    #define gpuMemGetInfo                   hipMemGetInfo
    #define gpuSetDevice                    hipSetDevice
    #define gpuGetDevice                    hipGetDevice
    #define gpuGetDeviceCount               hipGetDeviceCount
    #define gpuGetDeviceProperties          hipGetDeviceProperties
    #define gpuStreamCreate                 hipStreamCreate
    #define gpuStreamCreateWithFlags        hipStreamCreateWithFlags
    #define gpuStreamCreateWithPriority     hipStreamCreateWithPriority
    #define gpuStreamDestroy                hipStreamDestroy
    #define gpuStreamSynchronize            hipStreamSynchronize
    #define gpuStreamQuery                  hipStreamQuery
    #define gpuEventCreate                  hipEventCreate
    #define gpuEventCreateWithFlags         hipEventCreateWithFlags
    #define gpuEventDestroy                 hipEventDestroy
    #define gpuEventRecord                  hipEventRecord
    #define gpuEventSynchronize             hipEventSynchronize
    #define gpuEventElapsedTime             hipEventElapsedTime
    #define gpuEventQuery                   hipEventQuery  // ADD THIS LINE
    #define gpuGetLastError                 hipGetLastError
    #define gpuGetErrorString               hipGetErrorString
    #define gpuDeviceSynchronize            hipDeviceSynchronize
    #define gpuHostAlloc                    hipHostMalloc
    #define gpuFreeHost                     hipHostFree
    #define gpuDeviceSetLimit               hipDeviceSetLimit
    #define gpuDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange

    // Constants
    #define gpuSuccess                    hipSuccess
    #define gpuMemcpyHostToDevice         hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost         hipMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice       hipMemcpyDeviceToDevice
    #define gpuStreamNonBlocking          hipStreamNonBlocking
    #define gpuStreamDefault              hipStreamDefault
    #define gpuEventDisableTiming         hipEventDisableTiming
    #define gpuHostAllocMapped            hipHostMallocMapped
    #define gpuHostAllocWriteCombined     hipHostMallocWriteCombined
    #define gpuLimitPersistingL2CacheSize hipLimitPersistingL2CacheSize
    #define gpuErrorNotReady              hipErrorNotReady  // ADD THIS LINE

    // Device function qualifiers
    #define __gpu_device__      __device__
    #define __gpu_global__      __global__
    #define __gpu_host__        __host__
    #define __gpu_forceinline__ __forceinline__
    #define __gpu_shared__      __shared__
    #define __gpu_constant__    __constant__

    // Built-in functions
    #define __gpu_popc __popc
    #define __gpu_clz  __clz

#else
    // CUDA/NVIDIA GPU support
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>

// Type aliases for platform independence
using gpuError_t    = cudaError_t;
using gpuStream_t   = cudaStream_t;
using gpuEvent_t    = cudaEvent_t;
using gpuDeviceProp = cudaDeviceProp;
using gpuMemcpyKind = cudaMemcpyKind;

    // Function aliases
    #define gpuMalloc                       cudaMalloc
    #define gpuFree                         cudaFree
    #define gpuMemcpy                       cudaMemcpy
    #define gpuMemcpyAsync                  cudaMemcpyAsync
    #define gpuMemset                       cudaMemset
    #define gpuMemsetAsync                  cudaMemsetAsync
    #define gpuMemGetInfo                   cudaMemGetInfo
    #define gpuSetDevice                    cudaSetDevice
    #define gpuGetDevice                    cudaGetDevice
    #define gpuGetDeviceCount               cudaGetDeviceCount
    #define gpuGetDeviceProperties          cudaGetDeviceProperties
    #define gpuStreamCreate                 cudaStreamCreate
    #define gpuStreamCreateWithFlags        cudaStreamCreateWithFlags
    #define gpuStreamCreateWithPriority     cudaStreamCreateWithPriority
    #define gpuStreamDestroy                cudaStreamDestroy
    #define gpuStreamSynchronize            cudaStreamSynchronize
    #define gpuStreamQuery                  cudaStreamQuery
    #define gpuEventCreate                  cudaEventCreate
    #define gpuEventCreateWithFlags         cudaEventCreateWithFlags
    #define gpuEventDestroy                 cudaEventDestroy
    #define gpuEventRecord                  cudaEventRecord
    #define gpuEventSynchronize             cudaEventSynchronize
    #define gpuEventElapsedTime             cudaEventElapsedTime
    #define gpuEventQuery                   cudaEventQuery  // ADD THIS LINE
    #define gpuGetLastError                 cudaGetLastError
    #define gpuGetErrorString               cudaGetErrorString
    #define gpuDeviceSynchronize            cudaDeviceSynchronize
    #define gpuHostAlloc                    cudaHostAlloc
    #define gpuFreeHost                     cudaFreeHost
    #define gpuDeviceSetLimit               cudaDeviceSetLimit
    #define gpuDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange

    #define gpuHostAllocDefault           0
    #define gpuHostAllocPortable          cudaHostAllocPortable
    #define gpuHostAllocCoherent          cudaHostAllocDefault

    // Constants
    #define gpuSuccess                    cudaSuccess
    #define gpuMemcpyHostToDevice         cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost         cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice       cudaMemcpyDeviceToDevice
    #define gpuStreamNonBlocking          cudaStreamNonBlocking
    #define gpuStreamDefault              cudaStreamDefault
    #define gpuEventDisableTiming         cudaEventDisableTiming
    #define gpuHostAllocMapped            cudaHostAllocMapped
    #define gpuHostAllocWriteCombined     cudaHostAllocWriteCombined
    #define gpuLimitPersistingL2CacheSize cudaLimitPersistingL2CacheSize
    #define gpuErrorNotReady              cudaErrorNotReady  // ADD THIS LINE

    // Device function qualifiers
    #define __gpu_device__                __device__
    #define __gpu_global__                __global__
    #define __gpu_host__                  __host__
    #define __gpu_forceinline__           __forceinline__
    #define __gpu_shared__                __shared__
    #define __gpu_constant__              __constant__

    // Built-in functions
    #define __gpu_popc                    __popc
    #define __gpu_clz                     __clz

#endif

// Common GPU error checking macro
#define GPU_CHECK(call)                                                                                                \
    do {                                                                                                               \
        gpuError_t error = call;                                                                                       \
        if (error != gpuSuccess) {                                                                                     \
            fprintf(stderr, "GPU Error: %s at %s:%d\n", gpuGetErrorString(error), __FILE__, __LINE__);                 \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// Macro to ignore return value warnings
#define GPU_IGNORE_RESULT(call)                                                                                        \
    do {                                                                                                               \
        gpuError_t _ignored_result = call;                                                                             \
        (void)_ignored_result;                                                                                         \
    } while (0)

// Platform-specific includes for device code
#if defined(__CUDACC__) || defined(__HIPCC__)
    // Device code is being compiled
    #define GPU_DEVICE_CODE
#endif
