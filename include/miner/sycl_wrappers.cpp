#ifdef USE_SYCL

// Use more specific SYCL includes to avoid deprecated headers
#include <sycl/queue.hpp>
#include <sycl/device.hpp>
#include <sycl/context.hpp>
#include <sycl/platform.hpp>
#include <sycl/usm.hpp>
#include <sycl/property_list.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/aspects.hpp>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include "gpu_platform.hpp"

using namespace sycl;

// Global SYCL state
static std::unique_ptr<queue> g_sycl_queue = nullptr;
static std::unique_ptr<context> g_sycl_context = nullptr;
static std::unique_ptr<device> g_intel_device = nullptr;
static std::vector<void*> g_allocated_ptrs;

// Error codes
#define SYCL_SUCCESS 0
#define SYCL_ERROR_INVALID_VALUE 1
#define SYCL_ERROR_OUT_OF_MEMORY 2
#define SYCL_ERROR_NOT_INITIALIZED 3

// SYCL wrapper implementations
extern "C" {

gpuError_t gpuMalloc(void** ptr, size_t size) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        void* allocated = malloc_device(size, *g_sycl_queue);
        if (!allocated) {
            return SYCL_ERROR_OUT_OF_MEMORY;
        }
        *ptr = allocated;
        g_allocated_ptrs.push_back(allocated);
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

gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count, gpuMemcpyKind kind, gpuStream_t stream) {
    if (!g_sycl_queue) {
        return SYCL_ERROR_NOT_INITIALIZED;
    }

    try {
        // For now, use the global queue - in a full implementation, we'd use the stream parameter
        g_sycl_queue->memcpy(dst, src, count);
        return SYCL_SUCCESS;
    } catch (const sycl::exception& e) {
        return SYCL_ERROR_INVALID_VALUE;
    }
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
        auto devices = device::get_devices(info::device_type::gpu);
        *count = static_cast<int>(devices.size());
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

        prop->sharedMemPerBlock = 65536; // Estimate

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

        // Try to get actual L2 cache size
        try {
            auto local_mem_size = g_intel_device->get_info<info::device::local_mem_size>();
            prop->l2CacheSize = local_mem_size;
        } catch (...) {
            prop->l2CacheSize = 0; // Unknown
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
    *stream = g_sycl_queue.get();
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

// Initialize the SYCL wrapper system
extern "C" bool initialize_sycl_wrappers() {
    try {
        // Find Intel GPU device
        auto platforms = platform::get_platforms();
        device selected_device;
        bool found_intel_gpu = false;

        for (const auto& platform : platforms) {
            auto devices = platform.get_devices();
            for (const auto& device : devices) {
                if (device.is_gpu()) {
                    auto vendor = device.get_info<info::device::vendor>();
                    auto name = device.get_info<info::device::name>();

                    // Check for Intel GPU and basic compute capability
                    if ((vendor.find("Intel") != std::string::npos ||
                         name.find("Intel") != std::string::npos ||
                         name.find("Arc") != std::string::npos ||
                         name.find("Iris") != std::string::npos) &&
                        device.has(aspect::usm_device_allocations)) {
                        selected_device = device;
                        found_intel_gpu = true;
                        printf("Found Intel GPU: %s (Vendor: %s)\n", name.c_str(), vendor.c_str());
                        break;
                    }
                }
            }
            if (found_intel_gpu) break;
        }

        if (!found_intel_gpu) {
            printf("No Intel GPU found, falling back to any available GPU\n");
            auto devices = device::get_devices(info::device_type::gpu);
            if (!devices.empty()) {
                selected_device = devices[0];
                auto name = selected_device.get_info<info::device::name>();
                printf("Using GPU: %s\n", name.c_str());
            } else {
                printf("No GPU devices found\n");
                return false;
            }
        }

        // Create SYCL context and queue with modern API
        g_intel_device = std::make_unique<device>(selected_device);
        g_sycl_context = std::make_unique<context>(*g_intel_device);

        // Create queue with explicit device and enable profiling if supported
        property_list props;
        if (selected_device.has(aspect::queue_profiling)) {
            props = {property::queue::enable_profiling()};
        }
        g_sycl_queue = std::make_unique<queue>(*g_sycl_context, *g_intel_device, props);

        printf("SYCL wrappers initialized successfully\n");
        return true;

    } catch (const sycl::exception& e) {
        printf("SYCL exception during wrapper initialization: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        printf("Standard exception during SYCL wrapper initialization: %s\n", e.what());
        return false;
    }
}

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

    g_sycl_queue.reset();
    g_sycl_context.reset();
    g_intel_device.reset();
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

#endif // USE_SYCL