#include <iostream>

#include "sha1_miner.cuh"
#include "gpu_platform.hpp"

// Define the constant memory variable
__constant__ uint32_t d_base_message[8];

// Add this wrapper function
extern "C" void update_base_message_hip(const uint32_t* base_msg_words) {
    hipError_t err = hipMemcpyToSymbol(d_base_message, base_msg_words, 32);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to copy base message to constant memory: %s\n", hipGetErrorString(err));
    }
}

// SHA-1 constants
#define K0 0x5A827999
#define K1 0x6ED9EBA1
#define K2 0x8F1BBCDC
#define K3 0xCA62C1D6

#define H0_0 0x67452301
#define H0_1 0xEFCDAB89
#define H0_2 0x98BADCFE
#define H0_3 0x10325476
#define H0_4 0xC3D2E1F0

// AMD-specific optimizations
#define AMD_WAVEFRONT_SIZE 64  // For RDNA, runtime detection is better
#define LDS_BANK_CONFLICT_FREE_SIZE 33  // Avoid bank conflicts

/**
 * Optimized byte swap for AMD
 */
__device__ __forceinline__ uint32_t bswap32_amd(uint32_t x) {
    return __builtin_bswap32(x);
}

/**
 * AMD-optimized rotation using native rotate instruction
 */
__device__ __forceinline__ uint32_t amd_rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

/**
 * Count leading zero bits - fully unrolled like NVIDIA version
 */
__device__ __forceinline__ uint32_t count_leading_zeros_160bit(const uint32_t hash[5], const uint32_t target[5]) {
    uint32_t xor_val;
    uint32_t clz;

    xor_val = hash[0] ^ target[0];
    if (xor_val != 0) {
        clz = __builtin_clz(xor_val);
        return clz;
    }

    xor_val = hash[1] ^ target[1];
    if (xor_val != 0) {
        clz = __builtin_clz(xor_val);
        return 32 + clz;
    }

    xor_val = hash[2] ^ target[2];
    if (xor_val != 0) {
        clz = __builtin_clz(xor_val);
        return 64 + clz;
    }

    xor_val = hash[3] ^ target[3];
    if (xor_val != 0) {
        clz = __builtin_clz(xor_val);
        return 96 + clz;
    }

    xor_val = hash[4] ^ target[4];
    if (xor_val != 0) {
        clz = __builtin_clz(xor_val);
        return 128 + clz;
    }

    return 160;
}

// DUAL SHA-1 round macros for processing 2 hashes simultaneously (matching NVIDIA)
#define SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                   \
    do {                                                                                                               \
        uint32_t f1    = (b1 & c1) | (~b1 & d1);                                                                       \
        uint32_t f2    = (b2 & c2) | (~b2 & d2);                                                                       \
        uint32_t temp1 = amd_rotl32(a1, 5) + f1 + e1 + K0 + W1_val;                                                   \
        uint32_t temp2 = amd_rotl32(a2, 5) + f2 + e2 + K0 + W2_val;                                                   \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = amd_rotl32(b1, 30);                                                                           \
        c2             = amd_rotl32(b2, 30);                                                                           \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                  \
    do {                                                                                                               \
        uint32_t f1    = b1 ^ c1 ^ d1;                                                                                 \
        uint32_t f2    = b2 ^ c2 ^ d2;                                                                                 \
        uint32_t temp1 = amd_rotl32(a1, 5) + f1 + e1 + K1 + W1_val;                                                   \
        uint32_t temp2 = amd_rotl32(a2, 5) + f2 + e2 + K1 + W2_val;                                                   \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = amd_rotl32(b1, 30);                                                                           \
        c2             = amd_rotl32(b2, 30);                                                                           \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                  \
    do {                                                                                                               \
        uint32_t f1    = (b1 & c1) | (d1 & (b1 ^ c1));                                                                 \
        uint32_t f2    = (b2 & c2) | (d2 & (b2 ^ c2));                                                                 \
        uint32_t temp1 = amd_rotl32(a1, 5) + f1 + e1 + K2 + W1_val;                                                   \
        uint32_t temp2 = amd_rotl32(a2, 5) + f2 + e2 + K2 + W2_val;                                                   \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = amd_rotl32(b1, 30);                                                                           \
        c2             = amd_rotl32(b2, 30);                                                                           \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                  \
    do {                                                                                                               \
        uint32_t f1    = b1 ^ c1 ^ d1;                                                                                 \
        uint32_t f2    = b2 ^ c2 ^ d2;                                                                                 \
        uint32_t temp1 = amd_rotl32(a1, 5) + f1 + e1 + K3 + W1_val;                                                   \
        uint32_t temp2 = amd_rotl32(a2, 5) + f2 + e2 + K3 + W2_val;                                                   \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = amd_rotl32(b1, 30);                                                                           \
        c2             = amd_rotl32(b2, 30);                                                                           \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define COMPUTE_W_DUAL(t)                                                                                              \
    W1[t & 15] = amd_rotl32(W1[(t - 3) & 15] ^ W1[(t - 8) & 15] ^ W1[(t - 14) & 15] ^ W1[(t - 16) & 15], 1);          \
    W2[t & 15] = amd_rotl32(W2[(t - 3) & 15] ^ W2[(t - 8) & 15] ^ W2[(t - 14) & 15] ^ W2[(t - 16) & 15], 1)

/**
 * Main SHA-1 mining kernel for AMD GPUs - Now with dual hashing like NVIDIA
 */
__global__ void sha1_mining_kernel_amd(
    const uint32_t * __restrict__ target_hash,
    uint32_t difficulty,
    MiningResult * __restrict__ results,
    uint32_t * __restrict__ result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread,
    uint64_t job_version
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // Load target hash into registers
    uint32_t target[5];
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = target_hash[i];
    }

    // Process nonces in pairs - adjust loop to handle 2 at a time
    for (uint32_t i = 0; i < nonces_per_thread; i += 2) {
        uint64_t nonce1 = thread_nonce_base + i;
        uint64_t nonce2 = thread_nonce_base + i + 1;

        // Skip if either nonce is 0
        if (nonce1 == 0)
            nonce1 = thread_nonce_base + nonces_per_thread;
        if (nonce2 == 0)
            nonce2 = thread_nonce_base + nonces_per_thread + 1;

        // Create messages with both nonces
        uint32_t msg_words1[8], msg_words2[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            msg_words1[j] = d_base_message[j];
            msg_words2[j] = d_base_message[j];
        }

        // Apply nonces
        msg_words1[6] ^= bswap32_amd(nonce1 >> 32);
        msg_words1[7] ^= bswap32_amd(nonce1 & 0xFFFFFFFF);
        msg_words2[6] ^= bswap32_amd(nonce2 >> 32);
        msg_words2[7] ^= bswap32_amd(nonce2 & 0xFFFFFFFF);

        // Prepare W arrays for both hashes
        uint32_t W1[16], W2[16];

        // Unrolled byte swap for both
        W1[0] = bswap32_amd(msg_words1[0]);
        W2[0] = bswap32_amd(msg_words2[0]);
        W1[1] = bswap32_amd(msg_words1[1]);
        W2[1] = bswap32_amd(msg_words2[1]);
        W1[2] = bswap32_amd(msg_words1[2]);
        W2[2] = bswap32_amd(msg_words2[2]);
        W1[3] = bswap32_amd(msg_words1[3]);
        W2[3] = bswap32_amd(msg_words2[3]);
        W1[4] = bswap32_amd(msg_words1[4]);
        W2[4] = bswap32_amd(msg_words2[4]);
        W1[5] = bswap32_amd(msg_words1[5]);
        W2[5] = bswap32_amd(msg_words2[5]);
        W1[6] = bswap32_amd(msg_words1[6]);
        W2[6] = bswap32_amd(msg_words2[6]);
        W1[7] = bswap32_amd(msg_words1[7]);
        W2[7] = bswap32_amd(msg_words2[7]);

        // Apply padding to both
        W1[8]  = 0x80000000;
        W2[8]  = 0x80000000;
        W1[9]  = 0;
        W2[9]  = 0;
        W1[10] = 0;
        W2[10] = 0;
        W1[11] = 0;
        W2[11] = 0;
        W1[12] = 0;
        W2[12] = 0;
        W1[13] = 0;
        W2[13] = 0;
        W1[14] = 0;
        W2[14] = 0;
        W1[15] = 0x00000100;
        W2[15] = 0x00000100;

        // Initialize working variables for both hashes
        uint32_t a1 = H0_0, a2 = H0_0;
        uint32_t b1 = H0_1, b2 = H0_1;
        uint32_t c1 = H0_2, c2 = H0_2;
        uint32_t d1 = H0_3, d2 = H0_3;
        uint32_t e1 = H0_4, e2 = H0_4;

        // FULLY UNROLLED DUAL SHA-1 rounds (matching NVIDIA)
        // Rounds 0-15 (no message schedule needed)
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[0], a2, b2, c2, d2, e2, W2[0]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[1], a2, b2, c2, d2, e2, W2[1]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[2], a2, b2, c2, d2, e2, W2[2]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[3], a2, b2, c2, d2, e2, W2[3]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[4], a2, b2, c2, d2, e2, W2[4]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[5], a2, b2, c2, d2, e2, W2[5]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[6], a2, b2, c2, d2, e2, W2[6]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[7], a2, b2, c2, d2, e2, W2[7]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[8], a2, b2, c2, d2, e2, W2[8]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[9], a2, b2, c2, d2, e2, W2[9]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[10], a2, b2, c2, d2, e2, W2[10]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[11], a2, b2, c2, d2, e2, W2[11]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[12], a2, b2, c2, d2, e2, W2[12]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[13], a2, b2, c2, d2, e2, W2[13]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[14], a2, b2, c2, d2, e2, W2[14]);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[15], a2, b2, c2, d2, e2, W2[15]);

        // Rounds 16-19 with message schedule
        COMPUTE_W_DUAL(16);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[0], a2, b2, c2, d2, e2, W2[0]);
        COMPUTE_W_DUAL(17);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[1], a2, b2, c2, d2, e2, W2[1]);
        COMPUTE_W_DUAL(18);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[2], a2, b2, c2, d2, e2, W2[2]);
        COMPUTE_W_DUAL(19);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1[3], a2, b2, c2, d2, e2, W2[3]);

        // Rounds 20-39
        COMPUTE_W_DUAL(20);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[4], a2, b2, c2, d2, e2, W2[4]);
        COMPUTE_W_DUAL(21);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[5], a2, b2, c2, d2, e2, W2[5]);
        COMPUTE_W_DUAL(22);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[6], a2, b2, c2, d2, e2, W2[6]);
        COMPUTE_W_DUAL(23);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[7], a2, b2, c2, d2, e2, W2[7]);
        COMPUTE_W_DUAL(24);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[8], a2, b2, c2, d2, e2, W2[8]);
        COMPUTE_W_DUAL(25);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[9], a2, b2, c2, d2, e2, W2[9]);
        COMPUTE_W_DUAL(26);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[10], a2, b2, c2, d2, e2, W2[10]);
        COMPUTE_W_DUAL(27);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[11], a2, b2, c2, d2, e2, W2[11]);
        COMPUTE_W_DUAL(28);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[12], a2, b2, c2, d2, e2, W2[12]);
        COMPUTE_W_DUAL(29);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[13], a2, b2, c2, d2, e2, W2[13]);
        COMPUTE_W_DUAL(30);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[14], a2, b2, c2, d2, e2, W2[14]);
        COMPUTE_W_DUAL(31);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[15], a2, b2, c2, d2, e2, W2[15]);
        COMPUTE_W_DUAL(32);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[0], a2, b2, c2, d2, e2, W2[0]);
        COMPUTE_W_DUAL(33);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[1], a2, b2, c2, d2, e2, W2[1]);
        COMPUTE_W_DUAL(34);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[2], a2, b2, c2, d2, e2, W2[2]);
        COMPUTE_W_DUAL(35);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[3], a2, b2, c2, d2, e2, W2[3]);
        COMPUTE_W_DUAL(36);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[4], a2, b2, c2, d2, e2, W2[4]);
        COMPUTE_W_DUAL(37);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[5], a2, b2, c2, d2, e2, W2[5]);
        COMPUTE_W_DUAL(38);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[6], a2, b2, c2, d2, e2, W2[6]);
        COMPUTE_W_DUAL(39);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1[7], a2, b2, c2, d2, e2, W2[7]);

        // Rounds 40-59
        COMPUTE_W_DUAL(40);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[8], a2, b2, c2, d2, e2, W2[8]);
        COMPUTE_W_DUAL(41);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[9], a2, b2, c2, d2, e2, W2[9]);
        COMPUTE_W_DUAL(42);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[10], a2, b2, c2, d2, e2, W2[10]);
        COMPUTE_W_DUAL(43);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[11], a2, b2, c2, d2, e2, W2[11]);
        COMPUTE_W_DUAL(44);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[12], a2, b2, c2, d2, e2, W2[12]);
        COMPUTE_W_DUAL(45);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[13], a2, b2, c2, d2, e2, W2[13]);
        COMPUTE_W_DUAL(46);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[14], a2, b2, c2, d2, e2, W2[14]);
        COMPUTE_W_DUAL(47);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[15], a2, b2, c2, d2, e2, W2[15]);
        COMPUTE_W_DUAL(48);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[0], a2, b2, c2, d2, e2, W2[0]);
        COMPUTE_W_DUAL(49);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[1], a2, b2, c2, d2, e2, W2[1]);
        COMPUTE_W_DUAL(50);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[2], a2, b2, c2, d2, e2, W2[2]);
        COMPUTE_W_DUAL(51);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[3], a2, b2, c2, d2, e2, W2[3]);
        COMPUTE_W_DUAL(52);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[4], a2, b2, c2, d2, e2, W2[4]);
        COMPUTE_W_DUAL(53);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[5], a2, b2, c2, d2, e2, W2[5]);
        COMPUTE_W_DUAL(54);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[6], a2, b2, c2, d2, e2, W2[6]);
        COMPUTE_W_DUAL(55);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[7], a2, b2, c2, d2, e2, W2[7]);
        COMPUTE_W_DUAL(56);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[8], a2, b2, c2, d2, e2, W2[8]);
        COMPUTE_W_DUAL(57);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[9], a2, b2, c2, d2, e2, W2[9]);
        COMPUTE_W_DUAL(58);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[10], a2, b2, c2, d2, e2, W2[10]);
        COMPUTE_W_DUAL(59);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1[11], a2, b2, c2, d2, e2, W2[11]);

        // Rounds 60-79
        COMPUTE_W_DUAL(60);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[12], a2, b2, c2, d2, e2, W2[12]);
        COMPUTE_W_DUAL(61);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[13], a2, b2, c2, d2, e2, W2[13]);
        COMPUTE_W_DUAL(62);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[14], a2, b2, c2, d2, e2, W2[14]);
        COMPUTE_W_DUAL(63);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[15], a2, b2, c2, d2, e2, W2[15]);
        COMPUTE_W_DUAL(64);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[0], a2, b2, c2, d2, e2, W2[0]);
        COMPUTE_W_DUAL(65);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[1], a2, b2, c2, d2, e2, W2[1]);
        COMPUTE_W_DUAL(66);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[2], a2, b2, c2, d2, e2, W2[2]);
        COMPUTE_W_DUAL(67);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[3], a2, b2, c2, d2, e2, W2[3]);
        COMPUTE_W_DUAL(68);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[4], a2, b2, c2, d2, e2, W2[4]);
        COMPUTE_W_DUAL(69);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[5], a2, b2, c2, d2, e2, W2[5]);
        COMPUTE_W_DUAL(70);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[6], a2, b2, c2, d2, e2, W2[6]);
        COMPUTE_W_DUAL(71);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[7], a2, b2, c2, d2, e2, W2[7]);
        COMPUTE_W_DUAL(72);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[8], a2, b2, c2, d2, e2, W2[8]);
        COMPUTE_W_DUAL(73);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[9], a2, b2, c2, d2, e2, W2[9]);
        COMPUTE_W_DUAL(74);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[10], a2, b2, c2, d2, e2, W2[10]);
        COMPUTE_W_DUAL(75);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[11], a2, b2, c2, d2, e2, W2[11]);
        COMPUTE_W_DUAL(76);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[12], a2, b2, c2, d2, e2, W2[12]);
        COMPUTE_W_DUAL(77);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[13], a2, b2, c2, d2, e2, W2[13]);
        COMPUTE_W_DUAL(78);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[14], a2, b2, c2, d2, e2, W2[14]);
        COMPUTE_W_DUAL(79);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1[15], a2, b2, c2, d2, e2, W2[15]);

        // Final hash values for both
        uint32_t hash1[5], hash2[5];
        hash1[0] = a1 + H0_0;
        hash1[1] = b1 + H0_1;
        hash1[2] = c1 + H0_2;
        hash1[3] = d1 + H0_3;
        hash1[4] = e1 + H0_4;

        hash2[0] = a2 + H0_0;
        hash2[1] = b2 + H0_1;
        hash2[2] = c2 + H0_2;
        hash2[3] = d2 + H0_3;
        hash2[4] = e2 + H0_4;

        // Check difficulty for first hash
        uint32_t matching_bits1 = count_leading_zeros_160bit(hash1, target);
        if (matching_bits1 >= difficulty) {
            uint32_t idx = atomicAdd(result_count, 1);
            if (idx < result_capacity) {
                results[idx].nonce         = nonce1;
                results[idx].matching_bits = matching_bits1;
                results[idx].job_version   = job_version;
#pragma unroll
                for (int j = 0; j < 5; j++) {
                    results[idx].hash[j] = hash1[j];
                }
            }
        }

        // Check difficulty for second hash
        if (i + 1 < nonces_per_thread) {  // Make sure we're within bounds
            uint32_t matching_bits2 = count_leading_zeros_160bit(hash2, target);
            if (matching_bits2 >= difficulty) {
                uint32_t idx = atomicAdd(result_count, 1);
                if (idx < result_capacity) {
                    results[idx].nonce         = nonce2;
                    results[idx].matching_bits = matching_bits2;
                    results[idx].job_version   = job_version;
#pragma unroll
                    for (int j = 0; j < 5; j++) {
                        results[idx].hash[j] = hash2[j];
                    }
                }
            }
        }
    }
}

/**
 * Launch the optimized AMD HIP SHA-1 mining kernel - matching NVIDIA launcher
 */
extern "C" void launch_mining_kernel_amd(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
) {
    // Validate configuration
    if (config.blocks <= 0 || config.threads_per_block <= 0) {
        fprintf(stderr, "Invalid kernel configuration: blocks=%d, threads=%d\n",
                config.blocks, config.threads_per_block);
        return;
    }

    // Validate pool pointers
    if (!pool.results || !pool.count) {
        fprintf(stderr, "ERROR: Invalid pool pointers - results=%p, count=%p\n",
                pool.results, pool.count);
        return;
    }

    // Reset result count
    hipError_t err = hipMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to reset result count: %s\n", hipGetErrorString(err));
        return;
    }

    // Clear previous errors
    hipGetLastError();

    dim3 gridDim(config.blocks, 1, 1);
    dim3 blockDim(config.threads_per_block, 1, 1);

    hipLaunchKernelGGL(
        sha1_mining_kernel_amd,
        gridDim,
        blockDim,
        0,  // shared memory size
        config.stream,
        device_job.target_hash,
        difficulty,
        pool.results,
        pool.count,
        pool.capacity,
        nonce_offset,
        NONCES_PER_THREAD,
        job_version
    );

    // Check for launch errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(err));
    }
}