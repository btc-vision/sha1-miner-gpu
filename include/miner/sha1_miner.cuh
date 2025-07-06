#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Include CUDA device functions
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <sm_32_intrinsics.h>
#endif

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
    uint8_t base_message[32];     // Base message to modify
    uint32_t target_hash[5];      // Target hash we're trying to match
    uint32_t difficulty;          // Number of bits that must match
    uint64_t nonce_offset;        // Starting nonce for this job
};

// Result structure for found candidates
struct MiningResult {
    uint64_t nonce;              // The nonce that produced this result
    uint32_t hash[5];            // The resulting hash
    uint32_t matching_bits;      // Number of bits matching the target
    uint32_t difficulty_score;   // Additional difficulty metric
};

// GPU memory pool for results
struct ResultPool {
    MiningResult* results;       // Array of results
    uint32_t* count;            // Number of results found
    uint32_t capacity;          // Maximum number of results
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

// API functions
extern "C" {
    // Initialize mining system
    bool init_mining_system(int device_id);

    // Create a new mining job
    MiningJob create_mining_job(const uint8_t* message, const uint8_t* target_hash, uint32_t difficulty);

    // Launch mining kernel
    void launch_mining_kernel(const MiningJob& job, ResultPool& pool, const KernelConfig& config);

    // Process results
    int process_results(ResultPool& pool, MiningResult* host_results, int max_results);

    // Cleanup
    void cleanup_mining_system();
}

// Device functions for SHA-1 computation
#ifdef __CUDACC__

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
    // Use the CUDA intrinsic for funnel shift
    return __funnelshift_l(x, x, n);
}

__device__ __forceinline__ uint32_t swap_endian(uint32_t x) {
    // Use CUDA byte permutation intrinsic
    return __byte_perm(x, 0, 0x0123);
}

// Optimized bit counting for near-collision detection
__device__ __forceinline__ uint32_t count_matching_bits(uint32_t a, uint32_t b) {
    // Use CUDA population count intrinsic
    return 32 - __popc(a ^ b);
}

// Early exit check - can we already determine this won't meet difficulty?
__device__ __forceinline__ bool early_exit_check(
    const uint32_t* current_state,
    const uint32_t* target,
    uint32_t round,
    uint32_t required_bits
) {
    // Implement progressive checking based on rounds completed
    // This is a simplified version - can be made more sophisticated
    if (round >= 60) {
        // Check partial state matching
        uint32_t bits = 0;
        for (int i = 0; i < 5; i++) {
            bits += count_matching_bits(current_state[i], target[i]);
        }
        // If we're too far off with only 20 rounds left, exit early
        return bits < (required_bits - 40);
    }
    return false;
}

#endif // __CUDACC__