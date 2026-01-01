#include <cuda_runtime.h>

#include <cstdio>

#include "sha1_miner.cuh"

// Define the constant memory variables
__constant__ uint32_t d_base_message[8];
__constant__ uint32_t d_pre_swapped_base[8];

// CPU-side byte swap function
inline uint32_t bswap32_cpu(uint32_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    #include <stdlib.h>
    return _byteswap_ulong(x);
#else
    // Portable fallback
    return ((x & 0xFF000000) >> 24) | ((x & 0x00FF0000) >> 8) | ((x & 0x0000FF00) << 8) | ((x & 0x000000FF) << 24);
#endif
}

extern "C" void update_base_message_cuda(const uint32_t *base_msg_words)
{
    uint32_t pre_swapped[8];
    for (int j = 0; j < 8; j++) {
        pre_swapped[j] = bswap32_cpu(base_msg_words[j]);
    }

    // Use cudaGetSymbolAddress instead of cudaMemcpyToSymbol
    void *d_pre_swapped_ptr = nullptr;
    void *d_base_ptr = nullptr;

    cudaError_t err = cudaGetSymbolAddress(&d_pre_swapped_ptr, d_pre_swapped_base);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get d_pre_swapped_base address: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaGetSymbolAddress(&d_base_ptr, d_base_message);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get d_base_message address: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(d_pre_swapped_ptr, pre_swapped, sizeof(pre_swapped), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pre-swapped base message: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_base_ptr, base_msg_words, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy base message: %s\n", cudaGetErrorString(err));
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

// Optimized byte swap using PTX
__device__ __forceinline__ uint32_t bswap32_ptx(uint32_t x)
{
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
}

// Optimized count leading zeros - fully unrolled
__device__ __forceinline__ uint32_t count_leading_zeros_160bit(const uint32_t hash[5], const uint32_t target[5])
{
    uint32_t xor_val;
    uint32_t clz;

    xor_val = hash[0] ^ target[0];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return clz;
    }

    xor_val = hash[1] ^ target[1];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 32 + clz;
    }

    xor_val = hash[2] ^ target[2];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 64 + clz;
    }

    xor_val = hash[3] ^ target[3];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 96 + clz;
    }

    xor_val = hash[4] ^ target[4];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 128 + clz;
    }

    return 160;
}

// DUAL SHA-1 round macros for processing 2 hashes simultaneously
#define SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                   \
    do {                                                                                                               \
        uint32_t f1    = (b1 & c1) | (~b1 & d1);                                                                       \
        uint32_t f2    = (b2 & c2) | (~b2 & d2);                                                                       \
        uint32_t temp1 = __funnelshift_l(a1, a1, 5) + f1 + e1 + K0 + W1_val;                                           \
        uint32_t temp2 = __funnelshift_l(a2, a2, 5) + f2 + e2 + K0 + W2_val;                                           \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = __funnelshift_l(b1, b1, 30);                                                                  \
        c2             = __funnelshift_l(b2, b2, 30);                                                                  \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                  \
    do {                                                                                                               \
        uint32_t f1    = b1 ^ c1 ^ d1;                                                                                 \
        uint32_t f2    = b2 ^ c2 ^ d2;                                                                                 \
        uint32_t temp1 = __funnelshift_l(a1, a1, 5) + f1 + e1 + K1 + W1_val;                                           \
        uint32_t temp2 = __funnelshift_l(a2, a2, 5) + f2 + e2 + K1 + W2_val;                                           \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = __funnelshift_l(b1, b1, 30);                                                                  \
        c2             = __funnelshift_l(b2, b2, 30);                                                                  \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                  \
    do {                                                                                                               \
        uint32_t f1    = (b1 & c1) | (d1 & (b1 ^ c1));                                                                 \
        uint32_t f2    = (b2 & c2) | (d2 & (b2 ^ c2));                                                                 \
        uint32_t temp1 = __funnelshift_l(a1, a1, 5) + f1 + e1 + K2 + W1_val;                                           \
        uint32_t temp2 = __funnelshift_l(a2, a2, 5) + f2 + e2 + K2 + W2_val;                                           \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = __funnelshift_l(b1, b1, 30);                                                                  \
        c2             = __funnelshift_l(b2, b2, 30);                                                                  \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

#define SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val)                                  \
    do {                                                                                                               \
        uint32_t f1    = b1 ^ c1 ^ d1;                                                                                 \
        uint32_t f2    = b2 ^ c2 ^ d2;                                                                                 \
        uint32_t temp1 = __funnelshift_l(a1, a1, 5) + f1 + e1 + K3 + W1_val;                                           \
        uint32_t temp2 = __funnelshift_l(a2, a2, 5) + f2 + e2 + K3 + W2_val;                                           \
        e1             = d1;                                                                                           \
        e2             = d2;                                                                                           \
        d1             = c1;                                                                                           \
        d2             = c2;                                                                                           \
        c1             = __funnelshift_l(b1, b1, 30);                                                                  \
        c2             = __funnelshift_l(b2, b2, 30);                                                                  \
        b1             = a1;                                                                                           \
        b2             = a2;                                                                                           \
        a1             = temp1;                                                                                        \
        a2             = temp2;                                                                                        \
    } while (0)

// Message schedule macro adapted for scalars (using if for each index to avoid branches; since t is constant per call,
// compiler optimizes)
#define COMPUTE_W_DUAL(t)                                                                                              \
    do {                                                                                                               \
        uint32_t idx = (t) & 15;                                                                                       \
        if (idx == 0) {                                                                                                \
            W1_0 = __funnelshift_l(W1_13 ^ W1_8 ^ W1_2 ^ W1_0, W1_13 ^ W1_8 ^ W1_2 ^ W1_0, 1);                         \
            W2_0 = __funnelshift_l(W2_13 ^ W2_8 ^ W2_2 ^ W2_0, W2_13 ^ W2_8 ^ W2_2 ^ W2_0, 1);                         \
        } else if (idx == 1) {                                                                                         \
            W1_1 = __funnelshift_l(W1_14 ^ W1_9 ^ W1_3 ^ W1_1, W1_14 ^ W1_9 ^ W1_3 ^ W1_1, 1);                         \
            W2_1 = __funnelshift_l(W2_14 ^ W2_9 ^ W2_3 ^ W2_1, W2_14 ^ W2_9 ^ W2_3 ^ W2_1, 1);                         \
        } else if (idx == 2) {                                                                                         \
            W1_2 = __funnelshift_l(W1_15 ^ W1_10 ^ W1_4 ^ W1_2, W1_15 ^ W1_10 ^ W1_4 ^ W1_2, 1);                       \
            W2_2 = __funnelshift_l(W2_15 ^ W2_10 ^ W2_4 ^ W2_2, W2_15 ^ W2_10 ^ W2_4 ^ W2_2, 1);                       \
        } else if (idx == 3) {                                                                                         \
            W1_3 = __funnelshift_l(W1_0 ^ W1_11 ^ W1_5 ^ W1_3, W1_0 ^ W1_11 ^ W1_5 ^ W1_3, 1);                         \
            W2_3 = __funnelshift_l(W2_0 ^ W2_11 ^ W2_5 ^ W2_3, W2_0 ^ W2_11 ^ W2_5 ^ W2_3, 1);                         \
        } else if (idx == 4) {                                                                                         \
            W1_4 = __funnelshift_l(W1_1 ^ W1_12 ^ W1_6 ^ W1_4, W1_1 ^ W1_12 ^ W1_6 ^ W1_4, 1);                         \
            W2_4 = __funnelshift_l(W2_1 ^ W2_12 ^ W2_6 ^ W2_4, W2_1 ^ W2_12 ^ W2_6 ^ W2_4, 1);                         \
        } else if (idx == 5) {                                                                                         \
            W1_5 = __funnelshift_l(W1_2 ^ W1_13 ^ W1_7 ^ W1_5, W1_2 ^ W1_13 ^ W1_7 ^ W1_5, 1);                         \
            W2_5 = __funnelshift_l(W2_2 ^ W2_13 ^ W2_7 ^ W2_5, W2_2 ^ W2_13 ^ W2_7 ^ W2_5, 1);                         \
        } else if (idx == 6) {                                                                                         \
            W1_6 = __funnelshift_l(W1_3 ^ W1_14 ^ W1_8 ^ W1_6, W1_3 ^ W1_14 ^ W1_8 ^ W1_6, 1);                         \
            W2_6 = __funnelshift_l(W2_3 ^ W2_14 ^ W2_8 ^ W2_6, W2_3 ^ W2_14 ^ W2_8 ^ W2_6, 1);                         \
        } else if (idx == 7) {                                                                                         \
            W1_7 = __funnelshift_l(W1_4 ^ W1_15 ^ W1_9 ^ W1_7, W1_4 ^ W1_15 ^ W1_9 ^ W1_7, 1);                         \
            W2_7 = __funnelshift_l(W2_4 ^ W2_15 ^ W2_9 ^ W2_7, W2_4 ^ W2_15 ^ W2_9 ^ W2_7, 1);                         \
        } else if (idx == 8) {                                                                                         \
            W1_8 = __funnelshift_l(W1_5 ^ W1_0 ^ W1_10 ^ W1_8, W1_5 ^ W1_0 ^ W1_10 ^ W1_8, 1);                         \
            W2_8 = __funnelshift_l(W2_5 ^ W2_0 ^ W2_10 ^ W2_8, W2_5 ^ W2_0 ^ W2_10 ^ W2_8, 1);                         \
        } else if (idx == 9) {                                                                                         \
            W1_9 = __funnelshift_l(W1_6 ^ W1_1 ^ W1_11 ^ W1_9, W1_6 ^ W1_1 ^ W1_11 ^ W1_9, 1);                         \
            W2_9 = __funnelshift_l(W2_6 ^ W2_1 ^ W2_11 ^ W2_9, W2_6 ^ W2_1 ^ W2_11 ^ W2_9, 1);                         \
        } else if (idx == 10) {                                                                                        \
            W1_10 = __funnelshift_l(W1_7 ^ W1_2 ^ W1_12 ^ W1_10, W1_7 ^ W1_2 ^ W1_12 ^ W1_10, 1);                      \
            W2_10 = __funnelshift_l(W2_7 ^ W2_2 ^ W2_12 ^ W2_10, W2_7 ^ W2_2 ^ W2_12 ^ W2_10, 1);                      \
        } else if (idx == 11) {                                                                                        \
            W1_11 = __funnelshift_l(W1_8 ^ W1_3 ^ W1_13 ^ W1_11, W1_8 ^ W1_3 ^ W1_13 ^ W1_11, 1);                      \
            W2_11 = __funnelshift_l(W2_8 ^ W2_3 ^ W2_13 ^ W2_11, W2_8 ^ W2_3 ^ W2_13 ^ W2_11, 1);                      \
        } else if (idx == 12) {                                                                                        \
            W1_12 = __funnelshift_l(W1_9 ^ W1_4 ^ W1_14 ^ W1_12, W1_9 ^ W1_4 ^ W1_14 ^ W1_12, 1);                      \
            W2_12 = __funnelshift_l(W2_9 ^ W2_4 ^ W2_14 ^ W2_12, W2_9 ^ W2_4 ^ W2_14 ^ W2_12, 1);                      \
        } else if (idx == 13) {                                                                                        \
            W1_13 = __funnelshift_l(W1_10 ^ W1_5 ^ W1_15 ^ W1_13, W1_10 ^ W1_5 ^ W1_15 ^ W1_13, 1);                    \
            W2_13 = __funnelshift_l(W2_10 ^ W2_5 ^ W2_15 ^ W2_13, W2_10 ^ W2_5 ^ W2_15 ^ W2_13, 1);                    \
        } else if (idx == 14) {                                                                                        \
            W1_14 = __funnelshift_l(W1_11 ^ W1_6 ^ W1_0 ^ W1_14, W1_11 ^ W1_6 ^ W1_0 ^ W1_14, 1);                      \
            W2_14 = __funnelshift_l(W2_11 ^ W2_6 ^ W2_0 ^ W2_14, W2_11 ^ W2_6 ^ W2_0 ^ W2_14, 1);                      \
        } else if (idx == 15) {                                                                                        \
            W1_15 = __funnelshift_l(W1_12 ^ W1_7 ^ W1_1 ^ W1_15, W1_12 ^ W1_7 ^ W1_1 ^ W1_15, 1);                      \
            W2_15 = __funnelshift_l(W2_12 ^ W2_7 ^ W2_1 ^ W2_15, W2_12 ^ W2_7 ^ W2_1 ^ W2_15, 1);                      \
        }                                                                                                              \
    } while (0)

__global__ void sha1_mining_kernel_nvidia(const uint32_t *__restrict__ target_hash, uint32_t difficulty,
                                          MiningResult *__restrict__ results, uint32_t *__restrict__ result_count,
                                          uint32_t result_capacity, uint64_t nonce_base, uint32_t nonces_per_thread,
                                          uint64_t job_version)
{
    const uint32_t tid               = blockIdx.x * blockDim.x + threadIdx.x;
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

        // Scalar W variables for both hashes
        uint32_t W1_0, W1_1, W1_2, W1_3, W1_4, W1_5, W1_6, W1_7, W1_8, W1_9, W1_10, W1_11, W1_12, W1_13, W1_14, W1_15;
        uint32_t W2_0, W2_1, W2_2, W2_3, W2_4, W2_5, W2_6, W2_7, W2_8, W2_9, W2_10, W2_11, W2_12, W2_13, W2_14, W2_15;

        // Set fixed pre-swapped parts for 0-5 (same for both)
        W1_0 = d_pre_swapped_base[0];
        W2_0 = d_pre_swapped_base[0];
        W1_1 = d_pre_swapped_base[1];
        W2_1 = d_pre_swapped_base[1];
        W1_2 = d_pre_swapped_base[2];
        W2_2 = d_pre_swapped_base[2];
        W1_3 = d_pre_swapped_base[3];
        W2_3 = d_pre_swapped_base[3];
        W1_4 = d_pre_swapped_base[4];
        W2_4 = d_pre_swapped_base[4];
        W1_5 = d_pre_swapped_base[5];
        W2_5 = d_pre_swapped_base[5];

        // Set varying parts for 6-7 using pre-swapped base and direct nonce XOR (no bswap needed due to precompute)
        uint32_t nonce1_high = static_cast<uint32_t>(nonce1 >> 32);
        uint32_t nonce1_low  = static_cast<uint32_t>(nonce1 & 0xFFFFFFFF);
        W1_6                 = d_pre_swapped_base[6] ^ nonce1_high;
        W1_7                 = d_pre_swapped_base[7] ^ nonce1_low;

        uint32_t nonce2_high = static_cast<uint32_t>(nonce2 >> 32);
        uint32_t nonce2_low  = static_cast<uint32_t>(nonce2 & 0xFFFFFFFF);
        W2_6                 = d_pre_swapped_base[6] ^ nonce2_high;
        W2_7                 = d_pre_swapped_base[7] ^ nonce2_low;

        // Apply padding to both
        W1_8  = 0x80000000;
        W2_8  = 0x80000000;
        W1_9  = 0;
        W2_9  = 0;
        W1_10 = 0;
        W2_10 = 0;
        W1_11 = 0;
        W2_11 = 0;
        W1_12 = 0;
        W2_12 = 0;
        W1_13 = 0;
        W2_13 = 0;
        W1_14 = 0;
        W2_14 = 0;
        W1_15 = 0x00000100;
        W2_15 = 0x00000100;

        // Initialize working variables for both hashes
        uint32_t a1 = H0_0, a2 = H0_0;
        uint32_t b1 = H0_1, b2 = H0_1;
        uint32_t c1 = H0_2, c2 = H0_2;
        uint32_t d1 = H0_3, d2 = H0_3;
        uint32_t e1 = H0_4, e2 = H0_4;

        // FULLY UNROLLED DUAL SHA-1 rounds
        // Rounds 0-15 (no message schedule needed)
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);

        // Rounds 16-19 with message schedule
        COMPUTE_W_DUAL(16);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
        COMPUTE_W_DUAL(17);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
        COMPUTE_W_DUAL(18);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
        COMPUTE_W_DUAL(19);
        SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);

        // Rounds 20-39
        COMPUTE_W_DUAL(20);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
        COMPUTE_W_DUAL(21);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
        COMPUTE_W_DUAL(22);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
        COMPUTE_W_DUAL(23);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
        COMPUTE_W_DUAL(24);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
        COMPUTE_W_DUAL(25);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
        COMPUTE_W_DUAL(26);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
        COMPUTE_W_DUAL(27);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
        COMPUTE_W_DUAL(28);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
        COMPUTE_W_DUAL(29);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
        COMPUTE_W_DUAL(30);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
        COMPUTE_W_DUAL(31);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);
        COMPUTE_W_DUAL(32);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
        COMPUTE_W_DUAL(33);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
        COMPUTE_W_DUAL(34);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
        COMPUTE_W_DUAL(35);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
        COMPUTE_W_DUAL(36);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
        COMPUTE_W_DUAL(37);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
        COMPUTE_W_DUAL(38);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
        COMPUTE_W_DUAL(39);
        SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);

        // Rounds 40-59
        COMPUTE_W_DUAL(40);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
        COMPUTE_W_DUAL(41);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
        COMPUTE_W_DUAL(42);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
        COMPUTE_W_DUAL(43);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
        COMPUTE_W_DUAL(44);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
        COMPUTE_W_DUAL(45);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
        COMPUTE_W_DUAL(46);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
        COMPUTE_W_DUAL(47);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);
        COMPUTE_W_DUAL(48);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
        COMPUTE_W_DUAL(49);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
        COMPUTE_W_DUAL(50);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
        COMPUTE_W_DUAL(51);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
        COMPUTE_W_DUAL(52);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
        COMPUTE_W_DUAL(53);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
        COMPUTE_W_DUAL(54);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
        COMPUTE_W_DUAL(55);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
        COMPUTE_W_DUAL(56);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
        COMPUTE_W_DUAL(57);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
        COMPUTE_W_DUAL(58);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
        COMPUTE_W_DUAL(59);
        SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);

        // Rounds 60-79
        COMPUTE_W_DUAL(60);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
        COMPUTE_W_DUAL(61);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
        COMPUTE_W_DUAL(62);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
        COMPUTE_W_DUAL(63);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);
        COMPUTE_W_DUAL(64);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
        COMPUTE_W_DUAL(65);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
        COMPUTE_W_DUAL(66);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
        COMPUTE_W_DUAL(67);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
        COMPUTE_W_DUAL(68);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
        COMPUTE_W_DUAL(69);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
        COMPUTE_W_DUAL(70);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
        COMPUTE_W_DUAL(71);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
        COMPUTE_W_DUAL(72);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
        COMPUTE_W_DUAL(73);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
        COMPUTE_W_DUAL(74);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
        COMPUTE_W_DUAL(75);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
        COMPUTE_W_DUAL(76);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
        COMPUTE_W_DUAL(77);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
        COMPUTE_W_DUAL(78);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
        COMPUTE_W_DUAL(79);
        SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);

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

// Launcher function - remains the same
void launch_mining_kernel_nvidia(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                 const ResultPool &pool, const KernelConfig &config, uint64_t job_version)
{
    // Validate configuration
    if (config.blocks <= 0 || config.threads_per_block <= 0) {
        fprintf(stderr, "Invalid kernel configuration: blocks=%d, threads=%d\n", config.blocks,
                config.threads_per_block);
        return;
    }

    // Validate pool pointers
    if (!pool.results || !pool.count) {
        fprintf(stderr, "ERROR: Invalid pool pointers - results=%p, count=%p\n", pool.results, pool.count);
        return;
    }

    // Reset result count
    cudaError_t err = cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to reset result count: %s\n", cudaGetErrorString(err));
        return;
    }

    // Clear previous errors
    cudaGetLastError();

    dim3 gridDim(config.blocks, 1, 1);
    dim3 blockDim(config.threads_per_block, 1, 1);

    sha1_mining_kernel_nvidia<<<gridDim, blockDim, 0, config.stream>>>(device_job.target_hash, difficulty, pool.results,
                                                                       pool.count, pool.capacity, nonce_offset,
                                                                       NONCES_PER_THREAD, job_version);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
