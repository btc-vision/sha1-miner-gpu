#pragma once
#include "gpu_platform.hpp"
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

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
#define SHA1_BLOCK_SIZE 64
#define SHA1_DIGEST_SIZE 20
#define SHA1_ROUNDS 80

// Mining configuration - platform specific
#define MAX_CANDIDATES_PER_BATCH 1024

#ifdef USE_HIP
    // AMD GPUs need different values based on architecture
    // This will be overridden at runtime
    #define NONCES_PER_THREAD 100663296
    #define NONCES_PER_THREAD_RDNA1 24576
    #define NONCES_PER_THREAD_RDNA2 24576
    #define NONCES_PER_THREAD_RDNA3 32768
    #define NONCES_PER_THREAD_RDNA4 32768
    #define DEFAULT_THREADS_PER_BLOCK 512
#else
    #define NONCES_PER_THREAD 8192
    #define DEFAULT_THREADS_PER_BLOCK 512
#endif


// Debug mode flag
#define SHA1_MINER_DEBUG 0

// Difficulty is defined as the minimum number of matching bits required
struct MiningJob {
    uint8_t base_message[32]; // Base message to modify
    uint32_t target_hash[5]; // Target hash we're trying to match
    uint32_t difficulty; // Number of bits that must match
    uint64_t nonce_offset; // Starting nonce for this job
};

// Result structure for found candidates
struct MiningResult {
    uint64_t nonce; // The nonce that produced this result
    uint32_t hash[5]; // The resulting hash
    uint32_t matching_bits; // Number of bits matching the target
    uint32_t difficulty_score; // Additional difficulty metric
};

// GPU memory pool for results
struct ResultPool {
    MiningResult *results; // Array of results
    uint32_t *count; // Number of results found
    uint32_t capacity; // Maximum number of results
    uint64_t *nonces_processed;
};

// Mining statistics
struct MiningStats {
    uint64_t hashes_computed;
    uint64_t candidates_found;
    uint64_t best_match_bits;
    double hash_rate;
};

// Kernel launch parameters
struct KernelConfig {
    mutable int blocks;
    mutable int threads_per_block;
    int shared_memory_size;
    gpuStream_t stream;
};

// Device memory holder for mining job
struct DeviceMiningJob {
    uint8_t *base_message;
    uint32_t *target_hash;

    void allocate() {
        gpuError_t err = gpuMalloc(&base_message, 32);
        if (err != gpuSuccess) {
            fprintf(stderr, "Failed to allocate device memory for base_message: %s\n",
                    gpuGetErrorString(err));
            base_message = nullptr;
            return;
        }

        err = gpuMalloc(&target_hash, 5 * sizeof(uint32_t));
        if (err != gpuSuccess) {
            fprintf(stderr, "Failed to allocate device memory for target_hash: %s\n",
                    gpuGetErrorString(err));
            (void)gpuFree(base_message);  // Explicitly ignore return value
            base_message = nullptr;
            target_hash = nullptr;
        }
    }


    void free() {
        if (base_message) {
            gpuError_t err = gpuFree(base_message);
            if (err != gpuSuccess) {
                fprintf(stderr, "Warning: Failed to free base_message: %s\n", gpuGetErrorString(err));
            }
            base_message = nullptr;
        }
        if (target_hash) {
            gpuError_t err = gpuFree(target_hash);
            if (err != gpuSuccess) {
                fprintf(stderr, "Warning: Failed to free target_hash: %s\n", gpuGetErrorString(err));
            }
            target_hash = nullptr;
        }
    }

    void copyFromHost(const MiningJob &job) const {
        gpuError_t err = gpuMemcpy(base_message, job.base_message, 32, gpuMemcpyHostToDevice);
        if (err != gpuSuccess) {
            fprintf(stderr, "Failed to copy base_message to device: %s\n", gpuGetErrorString(err));
        }
        err = gpuMemcpy(target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice);
        if (err != gpuSuccess) {
            fprintf(stderr, "Failed to copy target_hash to device: %s\n", gpuGetErrorString(err));
        }
    }
};

// API functions - Host side
#ifdef __cplusplus
extern "C" {
#endif

// Initialize mining system
bool init_mining_system(int device_id);

// Create a new mining job
MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

// Cleanup
void cleanup_mining_system();

// Mining loop function
void run_mining_loop(MiningJob job);

#ifdef __cplusplus
}
#endif

// C++ only functions (not extern "C")
#ifdef __cplusplus

// Launch mining kernel - uses C++ references
void launch_mining_kernel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
);

#endif

// Device functions for SHA-1 computation
#if defined(__CUDACC__) || defined(__HIPCC__)

// Debug macros for device code
#if SHA1_MINER_DEBUG && (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
#define DEVICE_DEBUG_PRINT(...) \
    if (threadIdx.x == 0 && blockIdx.x == 0) { \
        printf("[DEVICE] " __VA_ARGS__); \
    }
#else
#define DEVICE_DEBUG_PRINT(...)
#endif

// Platform-optimized rotation function
__gpu_device__ __gpu_forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
#ifdef USE_HIP
    // AMD GPUs - use rotate intrinsic if available
    return __builtin_rotateleft32(x, n);
#else
// NVIDIA GPUs - use funnel shift on newer architectures
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __funnelshift_l(x, x, n);
#else
        return (x << n) | (x >> (32 - n));
#endif
#endif
}

// Platform-optimized endian swap
__gpu_device__ __gpu_forceinline__ uint32_t swap_endian(uint32_t x) {
#ifdef USE_HIP
    // AMD GPUs - use bswap intrinsic
    return __builtin_bswap32(x);
#else
// NVIDIA GPUs - use byte_perm on newer architectures
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
        return __byte_perm(x, 0, 0x0123);
#else
        return ((x & 0xFF000000) >> 24) |
               ((x & 0x00FF0000) >> 8)  |
               ((x & 0x0000FF00) << 8)  |
               ((x & 0x000000FF) << 24);
#endif
#endif
}

// Optimized bit counting for near-collision detection
__gpu_device__ __gpu_forceinline__ uint32_t count_matching_bits(uint32_t a, uint32_t b) {
    return 32 - __gpu_popc(a ^ b);
}

// Count leading zeros - platform optimized
__gpu_device__ __gpu_forceinline__ uint32_t count_leading_zeros(uint32_t x) {
    return __gpu_clz(x);
}

// Platform-specific memory prefetch
__gpu_device__ __gpu_forceinline__ void prefetch_data(const void* ptr) {
#ifdef USE_HIP
// AMD doesn't have explicit prefetch in the same way
// The compiler handles this automatically
#else
// CUDA doesn't have a direct prefetch instruction in device code
// The L1/L2 cache hierarchy handles this automatically
// We can use volatile loads to hint at cache usage
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
// Prefetch is handled by the cache hierarchy
// No explicit prefetch needed in CUDA device code
#endif
#endif
}

#endif // __CUDACC__ || __HIPCC__

// Helper functions for SHA-1 computation (host side) using OpenSSL
#if !defined(__CUDACC__) && !defined(__HIPCC__)

// Helper function to compute SHA-1 of binary data and return binary result
inline std::vector<uint8_t> sha1_binary(const uint8_t *data, size_t len) {
    std::vector<uint8_t> result(SHA_DIGEST_LENGTH);
    SHA1(data, len, result.data());
    return result;
}

// Helper function to compute SHA-1 of a vector
inline std::vector<uint8_t> sha1_binary(const std::vector<uint8_t> &data) {
    return sha1_binary(data.data(), data.size());
}

// Helper to convert binary hash to hex string
inline std::string sha1_hex(const uint8_t *hash) {
    std::ostringstream oss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(hash[i]);
    }
    return oss.str();
}

// Helper to convert hex string to binary
inline std::vector<uint8_t> hex_to_binary(const std::string &hex) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        result.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return result;
}

// Calculate SHA-1 (wrapper for consistency)
inline std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &data) {
    return sha1_binary(data);
}

// Convert bytes to hex string
inline std::string bytes_to_hex(const std::vector<uint8_t> &bytes) {
    std::ostringstream oss;
    for (uint8_t b: bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

// Convert hex string to bytes
inline std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    return hex_to_binary(hex);
}

#endif // !__CUDACC__ && !__HIPCC__
