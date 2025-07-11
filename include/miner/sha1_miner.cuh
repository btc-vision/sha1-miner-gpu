#pragma once
#include "gpu_platform.hpp"
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
#define MAX_CANDIDATES_PER_BATCH 1024

#ifdef USE_HIP
    #define NONCES_PER_THREAD 8192
    #define DEFAULT_THREADS_PER_BLOCK 512
#else
    #define NONCES_PER_THREAD 8192
    #define DEFAULT_THREADS_PER_BLOCK 512
#endif

struct MiningJob {
    uint8_t base_message[32];   // Base message to modify
    uint32_t target_hash[5];     // Target hash we're trying to match
    uint32_t difficulty;         // Number of bits that must match
    uint64_t nonce_offset;       // Starting nonce for this job
};

// Result structure for found candidates
struct MiningResult {
    uint64_t nonce;              // The nonce that produced this result
    uint32_t hash[5];            // The resulting hash
    uint32_t matching_bits;      // Number of bits matching the target
    uint32_t difficulty_score;   // Additional difficulty metric
    uint64_t job_version;        // Job version for this result
};

// GPU memory pool for results
struct ResultPool {
    MiningResult *results;       // Array of results
    uint32_t *count;            // Count of results found
    uint32_t capacity;          // Maximum results
    uint64_t *nonces_processed; // Total nonces processed
    uint64_t *job_version;      // Current job version (device memory)
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
    int blocks;
    int threads_per_block;
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
            gpuFree(base_message);
            base_message = nullptr;
            target_hash = nullptr;
        }
    }

    void free() {
        if (base_message) {
            gpuFree(base_message);
            base_message = nullptr;
        }
        if (target_hash) {
            gpuFree(target_hash);
            target_hash = nullptr;
        }
    }

    void copyFromHost(const MiningJob &job) const {
        gpuMemcpy(base_message, job.base_message, 32, gpuMemcpyHostToDevice);
        gpuMemcpy(target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice);
    }

    void updateFromHost(const MiningJob &job) const {
        copyFromHost(job); // Same as copy, but clearer intent
    }
};

// API functions - Host side
#ifdef __cplusplus
extern "C" {
#endif

bool init_mining_system(int device_id);
MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);
void cleanup_mining_system();
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
    const KernelConfig &config,
    uint64_t job_version
);

#endif // __cplusplus

// Device functions for SHA-1 computation
#if defined(__CUDACC__) || defined(__HIPCC__)

// Platform-optimized rotation function
__gpu_device__ __gpu_forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
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
__gpu_device__ __gpu_forceinline__ uint32_t swap_endian(uint32_t x) {
#ifdef USE_HIP
    return __builtin_bswap32(x);
#else
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

// Count leading zeros - platform optimized
__gpu_device__ __gpu_forceinline__ uint32_t count_leading_zeros(uint32_t x) {
    return __gpu_clz(x);
}

#endif // __CUDACC__ || __HIPCC__

// Helper functions for SHA-1 computation (host side) using OpenSSL
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline std::vector<uint8_t> sha1_binary(const uint8_t *data, size_t len) {
    std::vector<uint8_t> result(SHA_DIGEST_LENGTH);
    SHA1(data, len, result.data());
    return result;
}

inline std::vector<uint8_t> sha1_binary(const std::vector<uint8_t> &data) {
    return sha1_binary(data.data(), data.size());
}

inline std::string sha1_hex(const uint8_t *hash) {
    std::ostringstream oss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(hash[i]);
    }
    return oss.str();
}

inline std::vector<uint8_t> hex_to_binary(const std::string &hex) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        result.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return result;
}

inline std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &data) {
    return sha1_binary(data);
}

inline std::string bytes_to_hex(const std::vector<uint8_t> &bytes) {
    std::ostringstream oss;
    for (uint8_t b: bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

inline std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    return hex_to_binary(hex);
}

#endif // !__CUDACC__ && !__HIPCC__