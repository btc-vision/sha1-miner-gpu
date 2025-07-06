#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

#include "cxxsha1.hpp"

// SHA-1 constants
#define SHA1_BLOCK_SIZE 64
#define SHA1_DIGEST_SIZE 20
#define SHA1_ROUNDS 80

// Mining configuration
#define MAX_CANDIDATES_PER_BATCH 1024
#define NONCES_PER_THREAD 8
#define WARP_SIZE 32

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
    cudaStream_t stream;
};

// API functions - Host side
#ifdef __cplusplus
extern "C" {
#endif

// Initialize mining system
bool init_mining_system(int device_id);

// Create a new mining job
MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

// Launch mining kernel
void launch_mining_kernel(const MiningJob &job, ResultPool &pool, const KernelConfig &config);

// Cleanup
void cleanup_mining_system();

// Mining loop function
void run_mining_loop(MiningJob job, uint32_t duration_seconds);

#ifdef __cplusplus
}
#endif

// Device functions for SHA-1 computation
#ifdef __CUDACC__

// min() is already provided by CUDA math_functions.hpp

// Rotation function with architecture-specific optimization
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __funnelshift_l(x, x, n);
#else
    return (x << n) | (x >> (32 - n));
#endif
}

// Endian swap with architecture-specific optimization
__device__ __forceinline__ uint32_t swap_endian(uint32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return __byte_perm(x, 0, 0x0123);
#else
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
#endif
}

// Optimized bit counting for near-collision detection
__device__ __forceinline__ uint32_t count_matching_bits(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return 32 - __popc(a ^ b);
#else
    // Fallback implementation
    uint32_t x = a ^ b;
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0f0f0f0f;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return 32 - (x & 0x3f);
#endif
}

// Early exit check - can we already determine this won't meet difficulty?
__device__ __forceinline__ bool early_exit_check(
    const uint32_t* current_state,
    const uint32_t* target,
    uint32_t round,
    uint32_t required_bits
) {
    // Implement progressive checking based on rounds completed
    if (round >= 60) {
        // Check partial state matching
        uint32_t bits = 0;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            bits += count_matching_bits(current_state[i], target[i]);
        }
        // If we're too far off with only 20 rounds left, exit early
        return bits < (required_bits - 40);
    }
    return false;
}

#endif // __CUDACC__

// Helper functions for SHA-1 computation (host side)
#ifndef __CUDACC__

// Helper function to compute SHA-1 of binary data and return binary result
inline std::vector<uint8_t> sha1_binary(const uint8_t *data, size_t len) {
    SHA1 sha1;
    sha1.update(std::string(reinterpret_cast<const char *>(data), len));
    std::string hex = sha1.final();

    std::vector<uint8_t> result(20);
    for (int i = 0; i < 20; i++) {
        result[i] = static_cast<uint8_t>(std::stoi(hex.substr(i * 2, 2), nullptr, 16));
    }

    return result;
}

// Helper function to compute SHA-1 of a vector
inline std::vector<uint8_t> sha1_binary(const std::vector<uint8_t> &data) {
    return sha1_binary(data.data(), data.size());
}

// Helper to convert binary hash to hex string
inline std::string sha1_hex(const uint8_t *hash) {
    std::ostringstream oss;
    for (int i = 0; i < 20; i++) {
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

#endif // !__CUDACC__
