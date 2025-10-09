#ifdef USE_SYCL

// Use the full SYCL header like in main.cpp - this is proven to work
#include <sycl/sycl.hpp>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include "gpu_platform.hpp"
#include "sha1_miner_sycl.hpp"

using namespace sycl;

// External declarations for kernel functions
extern queue *g_sycl_queue;
extern context *g_sycl_context;
extern device *g_intel_device;

// Forward declarations
extern void sha1_mining_kernel_intel(
    queue& q,
    const uint32_t* target_hash,
    const uint32_t* pre_swapped_base,
    uint32_t difficulty,
    MiningResult* results,
    uint32_t* result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread,
    uint64_t job_version,
    int total_threads
);

// Local state for wrapper functions
static std::vector<void*> g_allocated_ptrs;

// Macros and functions are now defined in the included kernel file

// Forward declare functions from sha1_kernel_intel.sycl.cpp
extern "C" bool initialize_sycl_runtime();
extern "C" void cleanup_sycl_runtime();
extern "C" void update_base_message_sycl(const uint32_t *base_msg_words);
extern "C" void launch_mining_kernel_intel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
);


// Error codes
#define SYCL_SUCCESS 0
#define SYCL_ERROR_INVALID_VALUE 1
#define SYCL_ERROR_OUT_OF_MEMORY 2
#define SYCL_ERROR_NOT_INITIALIZED 3

// SYCL wrapper implementations
extern "C" {

    static std::unordered_set<void*> g_allocated_ptrs;
    static std::mutex g_alloc_mutex;  // Thread safety

gpuError_t gpuMalloc(void** ptr, size_t size) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        void* allocated = sycl::malloc_device(size, *g_sycl_queue);
        if (!allocated) {
            return SYCL_ERROR_OUT_OF_MEMORY;
        }
        *ptr = allocated;

        {
            std::lock_guard<std::mutex> lock(g_alloc_mutex);
            g_allocated_ptrs.insert(allocated);
        }
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_OUT_OF_MEMORY;
    }
}

gpuError_t gpuFree(void* ptr) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        free(ptr, *g_sycl_queue);
        auto it = std::find(g_allocated_ptrs.begin(), g_allocated_ptrs.end(), ptr);
        if (it != g_allocated_ptrs.end()) {
            g_allocated_ptrs.erase(it);
        }
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, gpuMemcpyKind kind) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        g_sycl_queue->memcpy(dst, src, count).wait();
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

static std::unordered_map<gpuStream_t, sycl::event> g_stream_events;

gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count,
                          gpuMemcpyKind kind, gpuStream_t stream) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        sycl::event e = g_sycl_queue->memcpy(dst, src, count);
        g_stream_events[stream] = e;  // Track the event
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
    auto it = g_stream_events.find(stream);
    if (it != g_stream_events.end()) {
        it->second.wait();  // Wait only when needed
    }
    return SYCL_SUCCESS;
}

gpuError_t gpuMemset(void* ptr, int value, size_t count) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        g_sycl_queue->memset(ptr, value, count).wait();
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuMemsetAsync(void* ptr, int value, size_t count, gpuStream_t stream) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        g_sycl_queue->memset(ptr, value, count);
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuMemGetInfo(size_t* free, size_t* total) {
    if (!g_intel_device) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        // SYCL doesn't provide direct memory info, so we'll estimate
        auto global_mem_size = g_intel_device->get_info<info::device::global_mem_size>();
        *total = global_mem_size;
        *free = global_mem_size * 0.8; // Estimate 80% available
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuSetDevice(int device) {
    // SYCL device selection is handled during initialization
    return SYCL_SUCCESS;
}

gpuError_t gpuGetDevice(int* device) {
    *device = 0; // Single device for now
    return SYCL_SUCCESS;
}

gpuError_t gpuGetDeviceCount(int* count) {
    try {
        // First check if SYCL runtime is initialized
        if (!g_intel_device || !g_sycl_queue) {
            *count = 0;
            return SYCL_ERROR_NOT_INITIALIZED;
        }

        // If runtime is initialized, we have at least 1 device
        *count = 1;
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        *count = 0;
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuGetDeviceProperties(gpuDeviceProp* prop, int device) {
    if (!g_intel_device) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        auto name = g_intel_device->get_info<info::device::name>();
        strncpy(prop->name, name.c_str(), sizeof(prop->name) - 1);
        prop->name[sizeof(prop->name) - 1] = '\0';

        prop->totalGlobalMem = g_intel_device->get_info<info::device::global_mem_size>();
        prop->major = 1;
        prop->minor = 0;

        auto compute_units = g_intel_device->get_info<info::device::max_compute_units>();
        prop->multiProcessorCount = static_cast<int>(compute_units);

        auto max_work_group_size = g_intel_device->get_info<info::device::max_work_group_size>();
        prop->maxThreadsPerBlock = static_cast<int>(max_work_group_size);

        auto max_work_item_sizes = g_intel_device->get_info<info::device::max_work_item_sizes<3>>();
        for (int i = 0; i < 3; i++) {
            prop->maxThreadsDim[i] = static_cast<int>(max_work_item_sizes[i]);
            prop->maxGridSize[i] = 65536; // Reasonable default
        }

        // Get actual shared/local memory size
        try {
            auto local_mem_size = g_intel_device->get_info<info::device::local_mem_size>();
            prop->sharedMemPerBlock = local_mem_size;
        } catch (...) {
            prop->sharedMemPerBlock = 65536; // Fallback estimate
        }

        // Try to get subgroup size, use fallback if not available
        try {
            auto sub_group_sizes = g_intel_device->get_info<info::device::sub_group_sizes>();
            if (!sub_group_sizes.empty()) {
                prop->warpSize = static_cast<int>(sub_group_sizes[0]);
            } else {
                prop->warpSize = 32; // Fallback
            }
        } catch (...) {
            prop->warpSize = 32; // Fallback if sub_group_sizes not supported
        }

        // Try to get actual clock rate and cache info
        try {
            if (g_intel_device->has(aspect::ext_intel_memory_clock_rate)) {
                prop->clockRate = g_intel_device->get_info<ext::intel::info::device::memory_clock_rate>() * 1000; // Convert to Hz
            } else {
                prop->clockRate = 1000000; // Fallback
            }
        } catch (...) {
            prop->clockRate = 1000000; // Fallback
        }


        // Set max threads per multiprocessor
        prop->maxThreadsPerMultiProcessor = static_cast<int>(max_work_group_size);

        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuStreamCreate(gpuStream_t* stream) {
    // For simplicity, return the global queue
    *stream = g_sycl_queue;
    return SYCL_SUCCESS;
}

gpuError_t gpuStreamCreateWithFlags(gpuStream_t* stream, unsigned int flags) {
    return gpuStreamCreate(stream);
}

gpuError_t gpuStreamDestroy(gpuStream_t stream) {
    // No-op for now since we're using the global queue
    return SYCL_SUCCESS;
}

gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        g_sycl_queue->wait();
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuStreamQuery(gpuStream_t stream) {
    // For simplicity, always return success (stream is ready)
    return SYCL_SUCCESS;
}

gpuError_t gpuEventCreate(gpuEvent_t* event) {
    // For simplicity, create a placeholder
    *event = malloc(sizeof(int));
    return SYCL_SUCCESS;
}

gpuError_t gpuEventDestroy(gpuEvent_t event) {
    if (event) {
        free(event);
    }
    return SYCL_SUCCESS;
}

gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
    // For simplicity, this is a no-op
    return SYCL_SUCCESS;
}

gpuError_t gpuEventSynchronize(gpuEvent_t event) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        g_sycl_queue->wait();
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end) {
    // For simplicity, return 0 (events are instant)
    *ms = 0.0f;
    return SYCL_SUCCESS;
}

gpuError_t gpuEventQuery(gpuEvent_t event) {
    // For simplicity, always return success (event is complete)
    return SYCL_SUCCESS;
}

static gpuError_t g_last_error = SYCL_SUCCESS;

gpuError_t gpuGetLastError(void) {
    gpuError_t error = g_last_error;
    g_last_error = SYCL_SUCCESS;
    return error;
}

const char* gpuGetErrorString(gpuError_t error) {
    switch (error) {
        case SYCL_SUCCESS: return "Success";
        case SYCL_ERROR_INVALID_VALUE: return "Invalid value";
        case SYCL_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case SYCL_ERROR_NOT_INITIALIZED: return "Not initialized";
        default: return "Unknown error";
    }
}

gpuError_t gpuDeviceSynchronize(void) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        g_sycl_queue->wait();
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
}

gpuError_t gpuHostAlloc(void** ptr, size_t size, unsigned int flags) {
    try {
        *ptr = malloc(size);
        return *ptr ? SYCL_SUCCESS : SYCL_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return SYCL_ERROR_OUT_OF_MEMORY;
    }
}

gpuError_t gpuFreeHost(void* ptr) {
    if (ptr) {
        free(ptr);
    }
    return SYCL_SUCCESS;
}

} // extern "C"


// Cleanup SYCL wrappers
extern "C" void cleanup_sycl_wrappers() {
    // Free any remaining allocated memory
    for (void* ptr : g_allocated_ptrs) {
        if (g_sycl_queue) {
            try {
                free(ptr, *g_sycl_queue);
            } catch (...) {
                // Ignore errors during cleanup
            }
        }
    }
    g_allocated_ptrs.clear();

    // Call the actual cleanup function from the kernel file
    cleanup_sycl_runtime();
}


// Additional functions for missing API compatibility
gpuError_t gpuDeviceReset(void) {
    // SYCL doesn't have an equivalent, but we can cleanup and reinitialize
    cleanup_sycl_runtime();
    return initialize_sycl_runtime() ? SYCL_SUCCESS : SYCL_ERROR_NOT_INITIALIZED;
}

gpuError_t gpuDeviceSetLimit(int limit, size_t value) {
    // SYCL doesn't have direct equivalent for device limits
    // Return success to maintain compatibility
    return SYCL_SUCCESS;
}

gpuError_t gpuEventCreateWithFlags(gpuEvent_t* event, unsigned int flags) {
    // For compatibility, just create a regular event
    return gpuEventCreate(event);
}

gpuError_t gpuDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    // SYCL doesn't have stream priorities like CUDA
    // Return default values for compatibility
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return SYCL_SUCCESS;
}

gpuError_t gpuStreamCreateWithPriority(gpuStream_t* stream, unsigned int flags, int priority) {
    // SYCL doesn't have stream priorities, just create a regular stream
    return gpuStreamCreateWithFlags(stream, flags);
}

// Initialize the SYCL wrapper system
extern "C" bool initialize_sycl_wrappers() {
    // Call the actual initialization from the kernel file first
    if (!initialize_sycl_runtime()) {
        return false;
    }

    printf("SYCL wrappers initialized successfully\n");
    return true;
}

#endif // USE_SYCL