#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// SHA-1 constants
#define SHA1_K0 0x5A827999u
#define SHA1_K1 0x6ED9EBA1u
#define SHA1_K2 0x8F1BBCDCu
#define SHA1_K3 0xCA62C1D6u

#define SHA1_H0 0x67452301u
#define SHA1_H1 0xEFCDAB89u
#define SHA1_H2 0x98BADCFEu
#define SHA1_H3 0x10325476u
#define SHA1_H4 0xC3D2E1F0u

// Mining configuration
#define CANDIDATES_RING_SIZE (1u << 16)  // 64K candidates
#define MIN_DIFFICULTY 20  // Minimum matching bits for near-collision
#define MAX_CANDIDATES_PER_BLOCK 32

// Difficulty is encoded as number of matching bits required
struct MiningJob {
    uint32_t base_msg[8];      // 32-byte base message
    uint32_t target_hash[5];   // Target SHA-1 hash
    uint32_t difficulty_bits;  // Number of bits that must match
    uint64_t nonce_start;      // Starting nonce for this job
    uint64_t nonce_range;      // Range to search
};

// Candidate structure for PCIe transfer
struct CollisionCandidate {
    uint64_t nonce;            // The nonce that produced this
    uint32_t hash[5];          // The resulting hash
    uint32_t distance_bits;    // Hamming distance in bits
    uint32_t thread_id;        // Which thread found it
};

// Job constants in constant memory
__device__ __constant__ MiningJob g_job;
__device__ __constant__ uint32_t g_bit_masks[32];  // Precomputed bit masks

// Inline PTX optimizations
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
    uint32_t result;
    asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(n));
    return result;
}

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
}

// Optimized bit counting using PTX
__device__ __forceinline__ uint32_t popcount32(uint32_t x) {
    uint32_t result;
    asm("popc.b32 %0, %1;" : "=r"(result) : "r"(x));
    return result;
}

// Calculate Hamming distance between two SHA-1 hashes
__device__ __forceinline__ uint32_t hamming_distance(const uint32_t h1[5], const uint32_t h2[5]) {
    uint32_t dist = 0;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        dist += popcount32(h1[i] ^ h2[i]);
    }
    return dist;
}

// Count matching bits from MSB
__device__ __forceinline__ uint32_t count_matching_bits(const uint32_t h1[5], const uint32_t h2[5]) {
    uint32_t matching = 0;

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        uint32_t xor_val = h1[i] ^ h2[i];
        if (xor_val == 0) {
            matching += 32;
        } else {
            // Count leading zeros (matching bits from MSB)
            uint32_t lz;
            asm("clz.b32 %0, %1;" : "=r"(lz) : "r"(xor_val));
            matching += lz;
            break;
        }
    }
    return matching;
}