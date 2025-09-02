#pragma once
#include <stdint.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_platform.hpp"

// Only include OpenSSL for host code
#if !defined(__CUDACC__) && !defined(__HIPCC__)
    #include <openssl/sha.h>
#else
    // Define SHA_DIGEST_LENGTH for device code if not already defined
    #ifndef SHA_DIGEST_LENGTH
        #define SHA_DIGEST_LENGTH 20
    #endif
#endif

// SHA-1 constants
#define SHA1_BLOCK_SIZE          64
#define SHA1_DIGEST_SIZE         20
#define SHA1_ROUNDS              80
#define MAX_CANDIDATES_PER_BATCH 1024

#ifdef USE_HIP
    #define NONCES_PER_THREAD         8192
    #define DEFAULT_THREADS_PER_BLOCK 256
#else
    #define NONCES_PER_THREAD         16384  // 32768 16384 131072
    #define DEFAULT_THREADS_PER_BLOCK 256
#endif

extern __constant__ uint32_t d_base_message[8];  // Always extern in header

struct alignas(256) MiningJob
{
    uint8_t base_message[32];  // Base message to modify
    uint32_t target_hash[5];   // Target hash we're trying to match
    uint32_t difficulty;       // Number of bits that must match
    uint64_t nonce_offset;     // Starting nonce for this job
};

// Result structure for found candidates
struct alignas(16) MiningResult
{
    uint64_t nonce;
    uint32_t hash[5];
    uint32_t matching_bits;
    uint32_t difficulty_score;
    uint64_t job_version;
    uint32_t padding[1];
};

// GPU memory pool for results
struct ResultPool
{
    MiningResult *results;  // Array of results
    uint32_t *count;        // Count of results found
    uint32_t capacity;      // Maximum results
    uint64_t *job_version;  // Current job version (device memory)
};

// Mining statistics
struct MiningStats
{
    uint64_t hashes_computed;
    uint64_t candidates_found;
    uint64_t best_match_bits;
    double hash_rate;
};

// Kernel launch parameters
struct KernelConfig
{
    int blocks;
    int threads_per_block;
    int shared_memory_size;
    gpuStream_t stream;
};

// Device memory holder for mining job
// Update in sha1_miner.cuh - replace the DeviceMiningJob struct with improved version

struct DeviceMiningJob
{
    uint32_t *target_hash;

    DeviceMiningJob() : target_hash(nullptr) {}

    ~DeviceMiningJob() { free(); }

    bool allocate()
    {
        // Free any existing allocations first
        free();

        // Check current device
        int current_device;
        gpuError_t err = gpuGetDevice(&current_device);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to get current device: %s\n", gpuGetErrorString(err));
            return false;
        }

        // Check available memory before allocation
        size_t free_mem, total_mem;
        err = gpuMemGetInfo(&free_mem, &total_mem);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to get memory info: %s\n", gpuGetErrorString(err));
        } else {
            fprintf(stderr, "[DeviceMiningJob] Device %d memory: %zu MB free / %zu MB total\n", current_device,
                    free_mem / (1024 * 1024), total_mem / (1024 * 1024));
        }

        // Define alignment requirements
        constexpr size_t alignment = 256;

        // Allocate base_message with proper alignment
        // Round up to alignment boundary
        constexpr size_t base_msg_aligned_size = ((32 + alignment - 1) / alignment) * alignment;
        fprintf(stderr, "[DeviceMiningJob] Allocating %zu bytes (aligned from 32) for base_message...\n",
                base_msg_aligned_size);

        void *temp_ptr = nullptr;
        err            = gpuMalloc(&temp_ptr, base_msg_aligned_size);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] ERROR: Failed to allocate base_message: %s (error code: %d)\n",
                    gpuGetErrorString(err), err);

            // Try to understand why allocation failed
#ifdef USE_HIP
            if (err == hipErrorOutOfMemory) {
                fprintf(stderr, "[DeviceMiningJob] Out of GPU memory\n");
            } else if (err == hipErrorInvalidValue) {
                fprintf(stderr, "[DeviceMiningJob] Invalid allocation size\n");
            } else if (err == hipErrorInvalidDevice) {
                fprintf(stderr, "[DeviceMiningJob] Invalid device\n");
            } else if (err == hipErrorNoDevice) {
                fprintf(stderr, "[DeviceMiningJob] No GPU device available\n");
            }
#else
            if (err == cudaErrorMemoryAllocation) {
                fprintf(stderr, "[DeviceMiningJob] Out of GPU memory\n");
            } else if (err == cudaErrorInvalidValue) {
                fprintf(stderr, "[DeviceMiningJob] Invalid allocation size\n");
            } else if (err == cudaErrorInvalidDevice) {
                fprintf(stderr, "[DeviceMiningJob] Invalid device\n");
            } else if (err == cudaErrorNoDevice) {
                fprintf(stderr, "[DeviceMiningJob] No CUDA device available\n");
            }
#endif
            return false;
        }
        // Allocate target_hash with proper alignment
        size_t target_size         = 5 * sizeof(uint32_t);  // 20 bytes
        size_t target_aligned_size = ((target_size + alignment - 1) / alignment) * alignment;
        fprintf(stderr, "[DeviceMiningJob] Allocating %zu bytes (aligned from %zu) for target_hash...\n",
                target_aligned_size, target_size);

        temp_ptr = nullptr;
        err      = gpuMalloc(&temp_ptr, target_aligned_size);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] ERROR: Failed to allocate target_hash: %s (error code: %d)\n",
                    gpuGetErrorString(err), err);
            target_hash = nullptr;
            return false;
        }

        target_hash = reinterpret_cast<uint32_t *>(temp_ptr);
        fprintf(stderr, "[DeviceMiningJob] Successfully allocated target_hash at %p (alignment: %zu bytes)\n",
                target_hash, (size_t)((uintptr_t)target_hash % alignment));

        // Verify alignment for target_hash
        if ((uintptr_t)target_hash % 16 != 0) {
            fprintf(stderr, "[DeviceMiningJob] WARNING: target_hash not 16-byte aligned for uint4 access!\n");
        }

        // Verify allocations
        if (!target_hash) {
            fprintf(stderr, "[DeviceMiningJob] ERROR: Pointers are null after allocation\n");
            free();
            return false;
        }

        err = gpuMemset(target_hash, 0, target_aligned_size);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Warning: Failed to clear target_hash: %s\n", gpuGetErrorString(err));
        }

        // Final alignment verification
        fprintf(stderr, "[DeviceMiningJob] Allocation successful - Final alignment check:\n");
        fprintf(stderr, "  target_hash: %p (mod 16: %zu, mod 256: %zu)\n", target_hash,
                (size_t)((uintptr_t)target_hash % 16), (size_t)((uintptr_t)target_hash % 256));

        return true;
    }

    void free()
    {
        if (target_hash) {
            gpuError_t err = gpuFree(target_hash);
            if (err != gpuSuccess) {
                fprintf(stderr, "[DeviceMiningJob] Warning: Failed to free target_hash: %s\n", gpuGetErrorString(err));
            }
            target_hash = nullptr;
        }
    }

    void copyFromHost(const MiningJob &job) const
    {
        if (!target_hash) {
            fprintf(stderr, "[DeviceMiningJob] Error: Not allocated (target_hash=%p)\n", target_hash);
            return;
        }

        gpuError_t err = gpuMemcpy(target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to copy target_hash: %s\n", gpuGetErrorString(err));
        }
    }

    void copyFromHostAsync(const MiningJob &job, gpuStream_t stream) const
    {
        if (!target_hash) {
            fprintf(stderr, "[DeviceMiningJob] Error: Not allocated\n");
            return;
        }

        gpuError_t err =
            gpuMemcpyAsync(target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice, stream);
        if (err != gpuSuccess) {
            fprintf(stderr, "[DeviceMiningJob] Failed to async copy target_hash: %s\n", gpuGetErrorString(err));
        }
    }

    void updateFromHost(const MiningJob &job) const { copyFromHost(job); }

    bool isAllocated() const { return target_hash != nullptr; }
};

// API functions - Host side
#ifdef __cplusplus
extern "C"
{
#endif

    MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

    void cleanup_mining_system();

    void run_mining_loop(const MiningJob &job);

#ifdef __cplusplus
}
#endif

// C++ only functions (not extern "C")
#ifdef __cplusplus

// Launch mining kernel - uses C++ references
void launch_mining_kernel(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                          const ResultPool &pool, const KernelConfig &config, uint64_t job_version);

#endif  // __cplusplus

// Device functions for SHA-1 computation
#if defined(__CUDACC__) || defined(__HIPCC__)

// Platform-optimized rotation function
__gpu_device__ __gpu_forceinline__ uint32_t rotl32(uint32_t x, uint32_t n)
{
    #ifdef USE_HIP
    return __builtin_rotateleft32(x, n);
    #else
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __funnelshift_l(x, x, n);
        #else
    return (x << n) | (x >> (32 - n));
        #endif
    #endif
}

// Platform-optimized endian swap
__gpu_device__ __gpu_forceinline__ uint32_t swap_endian(uint32_t x)
{
    #ifdef USE_HIP
    return __builtin_bswap32(x);
    #else
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return __byte_perm(x, 0, 0x0123);
        #else
    return ((x & 0xFF000000) >> 24) | ((x & 0x00FF0000) >> 8) | ((x & 0x0000FF00) << 8) | ((x & 0x000000FF) << 24);
        #endif
    #endif
}

// Count leading zeros - platform optimized
__gpu_device__ __gpu_forceinline__ uint32_t count_leading_zeros(uint32_t x)
{
    return __gpu_clz(x);
}

#endif  // __CUDACC__ || __HIPCC__

// Helper functions for SHA-1 computation (host side) using OpenSSL
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline std::vector<uint8_t> sha1_binary(const uint8_t *data, size_t len)
{
    std::vector<uint8_t> result(SHA_DIGEST_LENGTH);
    SHA1(data, len, result.data());
    return result;
}

inline std::vector<uint8_t> sha1_binary(const std::vector<uint8_t> &data)
{
    return sha1_binary(data.data(), data.size());
}

inline std::string sha1_hex(const uint8_t *hash)
{
    std::ostringstream oss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

inline std::vector<uint8_t> hex_to_binary(const std::string &hex)
{
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        result.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return result;
}

inline std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &data)
{
    return sha1_binary(data);
}

inline std::string bytes_to_hex(const std::vector<uint8_t> &bytes)
{
    std::ostringstream oss;
    for (uint8_t b : bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

inline std::vector<uint8_t> hex_to_bytes(const std::string &hex)
{
    return hex_to_binary(hex);
}

#endif  // !__CUDACC__ && !__HIPCC__
