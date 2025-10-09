#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <cstdio>
#include <cstring>
#include "sha1_miner_sycl.hpp"

using namespace sycl;

// Global SYCL queue and context for Intel GPU mining
queue *g_sycl_queue = nullptr;
context *g_sycl_context = nullptr;
device *g_intel_device = nullptr;

// Global constant memory for base message (SYCL equivalent)
static uint32_t *d_base_message_sycl = nullptr;
static uint32_t *d_pre_swapped_base_sycl = nullptr;
static uint32_t *d_target_hash_sycl = nullptr;

// SHA-1 constants
constexpr uint32_t K0 = 0x5A827999;
constexpr uint32_t K1 = 0x6ED9EBA1;
constexpr uint32_t K2 = 0x8F1BBCDC;
constexpr uint32_t K3 = 0xCA62C1D6;

constexpr uint32_t H0_0 = 0x67452301;
constexpr uint32_t H0_1 = 0xEFCDAB89;
constexpr uint32_t H0_2 = 0x98BADCFE;
constexpr uint32_t H0_3 = 0x10325476;
constexpr uint32_t H0_4 = 0xC3D2E1F0;

// Intel GPU optimized byte swap
inline uint32_t intel_bswap32(uint32_t x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8) |
           ((x & 0x0000FF00) << 8) |
           ((x & 0x000000FF) << 24);
}

// Intel GPU optimized rotation - manual implementation for compatibility
inline uint32_t intel_rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// Intel GPU optimized count leading zeros using SYCL built-in
inline uint32_t intel_clz(uint32_t x) {
    return sycl::clz(x);
}

// Count leading zeros for 160-bit comparison
inline uint32_t count_leading_zeros_160bit_intel(const uint32_t hash[5], const uint32_t target[5]) {
    uint32_t xor_val;
    uint32_t clz;

    xor_val = hash[0] ^ target[0];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return clz;
    }

    xor_val = hash[1] ^ target[1];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 32 + clz;
    }

    xor_val = hash[2] ^ target[2];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 64 + clz;
    }

    xor_val = hash[3] ^ target[3];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 96 + clz;
    }

    xor_val = hash[4] ^ target[4];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 128 + clz;
    }

    return 160;
}

// DUAL SHA-1 round macros for processing 2 hashes simultaneously
#define SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val) \
    do { \
        uint32_t f1 = (b1 & c1) | (~b1 & d1); \
        uint32_t f2 = (b2 & c2) | (~b2 & d2); \
        uint32_t temp1 = intel_rotl32(a1, 5) + f1 + e1 + K0 + W1_val; \
        uint32_t temp2 = intel_rotl32(a2, 5) + f2 + e2 + K0 + W2_val; \
        e1 = d1; \
        e2 = d2; \
        d1 = c1; \
        d2 = c2; \
        c1 = intel_rotl32(b1, 30); \
        c2 = intel_rotl32(b2, 30); \
        b1 = a1; \
        b2 = a2; \
        a1 = temp1; \
        a2 = temp2; \
    } while (0)

#define SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val) \
    do { \
        uint32_t f1 = b1 ^ c1 ^ d1; \
        uint32_t f2 = b2 ^ c2 ^ d2; \
        uint32_t temp1 = intel_rotl32(a1, 5) + f1 + e1 + K1 + W1_val; \
        uint32_t temp2 = intel_rotl32(a2, 5) + f2 + e2 + K1 + W2_val; \
        e1 = d1; \
        e2 = d2; \
        d1 = c1; \
        d2 = c2; \
        c1 = intel_rotl32(b1, 30); \
        c2 = intel_rotl32(b2, 30); \
        b1 = a1; \
        b2 = a2; \
        a1 = temp1; \
        a2 = temp2; \
    } while (0)

#define SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val) \
    do { \
        uint32_t f1 = (b1 & c1) | (d1 & (b1 ^ c1)); \
        uint32_t f2 = (b2 & c2) | (d2 & (b2 ^ c2)); \
        uint32_t temp1 = intel_rotl32(a1, 5) + f1 + e1 + K2 + W1_val; \
        uint32_t temp2 = intel_rotl32(a2, 5) + f2 + e2 + K2 + W2_val; \
        e1 = d1; \
        e2 = d2; \
        d1 = c1; \
        d2 = c2; \
        c1 = intel_rotl32(b1, 30); \
        c2 = intel_rotl32(b2, 30); \
        b1 = a1; \
        b2 = a2; \
        a1 = temp1; \
        a2 = temp2; \
    } while (0)

#define SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_val, a2, b2, c2, d2, e2, W2_val) \
    do { \
        uint32_t f1 = b1 ^ c1 ^ d1; \
        uint32_t f2 = b2 ^ c2 ^ d2; \
        uint32_t temp1 = intel_rotl32(a1, 5) + f1 + e1 + K3 + W1_val; \
        uint32_t temp2 = intel_rotl32(a2, 5) + f2 + e2 + K3 + W2_val; \
        e1 = d1; \
        e2 = d2; \
        d1 = c1; \
        d2 = c2; \
        c1 = intel_rotl32(b1, 30); \
        c2 = intel_rotl32(b2, 30); \
        b1 = a1; \
        b2 = a2; \
        a1 = temp1; \
        a2 = temp2; \
    } while (0)

// Message schedule macro adapted for scalars
#define COMPUTE_W_DUAL(t) \
    do { \
        uint32_t idx = (t) & 15; \
        if (idx == 0) { \
            W1_0 = intel_rotl32(W1_13 ^ W1_8 ^ W1_2 ^ W1_0, 1); \
            W2_0 = intel_rotl32(W2_13 ^ W2_8 ^ W2_2 ^ W2_0, 1); \
        } else if (idx == 1) { \
            W1_1 = intel_rotl32(W1_14 ^ W1_9 ^ W1_3 ^ W1_1, 1); \
            W2_1 = intel_rotl32(W2_14 ^ W2_9 ^ W2_3 ^ W2_1, 1); \
        } else if (idx == 2) { \
            W1_2 = intel_rotl32(W1_15 ^ W1_10 ^ W1_4 ^ W1_2, 1); \
            W2_2 = intel_rotl32(W2_15 ^ W2_10 ^ W2_4 ^ W2_2, 1); \
        } else if (idx == 3) { \
            W1_3 = intel_rotl32(W1_0 ^ W1_11 ^ W1_5 ^ W1_3, 1); \
            W2_3 = intel_rotl32(W2_0 ^ W2_11 ^ W2_5 ^ W2_3, 1); \
        } else if (idx == 4) { \
            W1_4 = intel_rotl32(W1_1 ^ W1_12 ^ W1_6 ^ W1_4, 1); \
            W2_4 = intel_rotl32(W2_1 ^ W2_12 ^ W2_6 ^ W2_4, 1); \
        } else if (idx == 5) { \
            W1_5 = intel_rotl32(W1_2 ^ W1_13 ^ W1_7 ^ W1_5, 1); \
            W2_5 = intel_rotl32(W2_2 ^ W2_13 ^ W2_7 ^ W2_5, 1); \
        } else if (idx == 6) { \
            W1_6 = intel_rotl32(W1_3 ^ W1_14 ^ W1_8 ^ W1_6, 1); \
            W2_6 = intel_rotl32(W2_3 ^ W2_14 ^ W2_8 ^ W2_6, 1); \
        } else if (idx == 7) { \
            W1_7 = intel_rotl32(W1_4 ^ W1_15 ^ W1_9 ^ W1_7, 1); \
            W2_7 = intel_rotl32(W2_4 ^ W2_15 ^ W2_9 ^ W2_7, 1); \
        } else if (idx == 8) { \
            W1_8 = intel_rotl32(W1_5 ^ W1_0 ^ W1_10 ^ W1_8, 1); \
            W2_8 = intel_rotl32(W2_5 ^ W2_0 ^ W2_10 ^ W2_8, 1); \
        } else if (idx == 9) { \
            W1_9 = intel_rotl32(W1_6 ^ W1_1 ^ W1_11 ^ W1_9, 1); \
            W2_9 = intel_rotl32(W2_6 ^ W2_1 ^ W2_11 ^ W2_9, 1); \
        } else if (idx == 10) { \
            W1_10 = intel_rotl32(W1_7 ^ W1_2 ^ W1_12 ^ W1_10, 1); \
            W2_10 = intel_rotl32(W2_7 ^ W2_2 ^ W2_12 ^ W2_10, 1); \
        } else if (idx == 11) { \
            W1_11 = intel_rotl32(W1_8 ^ W1_3 ^ W1_13 ^ W1_11, 1); \
            W2_11 = intel_rotl32(W2_8 ^ W2_3 ^ W2_13 ^ W2_11, 1); \
        } else if (idx == 12) { \
            W1_12 = intel_rotl32(W1_9 ^ W1_4 ^ W1_14 ^ W1_12, 1); \
            W2_12 = intel_rotl32(W2_9 ^ W2_4 ^ W2_14 ^ W2_12, 1); \
        } else if (idx == 13) { \
            W1_13 = intel_rotl32(W1_10 ^ W1_5 ^ W1_15 ^ W1_13, 1); \
            W2_13 = intel_rotl32(W2_10 ^ W2_5 ^ W2_15 ^ W2_13, 1); \
        } else if (idx == 14) { \
            W1_14 = intel_rotl32(W1_11 ^ W1_6 ^ W1_0 ^ W1_14, 1); \
            W2_14 = intel_rotl32(W2_11 ^ W2_6 ^ W2_0 ^ W2_14, 1); \
        } else if (idx == 15) { \
            W1_15 = intel_rotl32(W1_12 ^ W1_7 ^ W1_1 ^ W1_15, 1); \
            W2_15 = intel_rotl32(W2_12 ^ W2_7 ^ W2_1 ^ W2_15, 1); \
        } \
    } while (0)

// Intel GPU SHA-1 kernel using SYCL with per-thread result collection to prevent corruption
sycl::event sha1_mining_kernel_intel(
    queue& q,
    const uint32_t* target_hash_device,
    const uint32_t* pre_swapped_base,
    uint32_t difficulty,
    MiningResult* results,
    uint32_t* result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread,
    uint64_t job_ver,
    int total_threads
) {
    // Allocate temporary buffer for thread-local results to avoid race conditions
    constexpr int MAX_RESULTS_PER_THREAD = 8;
    MiningResult* temp_results = malloc_device<MiningResult>(total_threads * MAX_RESULTS_PER_THREAD, q);
    uint32_t* thread_counts = malloc_device<uint32_t>(total_threads, q);

    if (!temp_results || !thread_counts) {
        if (temp_results) free(temp_results, q);
        if (thread_counts) free(thread_counts, q);
        return q.submit([=](handler& h) { h.single_task([=]() {}); }); // Return empty event
    }

    // Clear thread counts AND temporary results buffer
    q.memset(thread_counts, 0, total_threads * sizeof(uint32_t)).wait();
    q.memset(temp_results, 0, total_threads * MAX_RESULTS_PER_THREAD * sizeof(MiningResult)).wait();

    // Launch main kernel with per-thread collection
    auto kernel_event = q.submit([=](handler& h) {
        h.parallel_for(range<1>(total_threads), [=](id<1> idx) {
            const uint32_t tid = idx[0];
            const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

            // Thread-local result buffer (no race conditions here!)
            MiningResult thread_results[MAX_RESULTS_PER_THREAD];
            uint32_t thread_result_count = 0;

            // Load target hash into registers from USM pointer
            uint32_t target[5];
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                target[i] = target_hash_device[i];
            }

        // Process nonces in pairs - but stop if thread buffer is full
        for (uint32_t i = 0; i < nonces_per_thread && thread_result_count < MAX_RESULTS_PER_THREAD; i += 2) {
            uint64_t nonce1 = thread_nonce_base + i;
            uint64_t nonce2 = thread_nonce_base + i + 1;

            // Skip if either nonce is 0
            if (nonce1 == 0) nonce1 = thread_nonce_base + nonces_per_thread;
            if (nonce2 == 0) nonce2 = thread_nonce_base + nonces_per_thread + 1;

            // Scalar W variables for both hashes
            uint32_t W1_0, W1_1, W1_2, W1_3, W1_4, W1_5, W1_6, W1_7, W1_8, W1_9, W1_10, W1_11, W1_12, W1_13, W1_14, W1_15;
            uint32_t W2_0, W2_1, W2_2, W2_3, W2_4, W2_5, W2_6, W2_7, W2_8, W2_9, W2_10, W2_11, W2_12, W2_13, W2_14, W2_15;

            // Set fixed pre-swapped parts for 0-5 (same for both)
            W1_0 = pre_swapped_base[0]; W2_0 = pre_swapped_base[0];
            W1_1 = pre_swapped_base[1]; W2_1 = pre_swapped_base[1];
            W1_2 = pre_swapped_base[2]; W2_2 = pre_swapped_base[2];
            W1_3 = pre_swapped_base[3]; W2_3 = pre_swapped_base[3];
            W1_4 = pre_swapped_base[4]; W2_4 = pre_swapped_base[4];
            W1_5 = pre_swapped_base[5]; W2_5 = pre_swapped_base[5];

            // Set varying parts for 6-7 using pre-swapped base and direct nonce XOR
            uint32_t nonce1_high = static_cast<uint32_t>(nonce1 >> 32);
            uint32_t nonce1_low = static_cast<uint32_t>(nonce1 & 0xFFFFFFFF);
            W1_6 = pre_swapped_base[6] ^ nonce1_high;
            W1_7 = pre_swapped_base[7] ^ nonce1_low;

            uint32_t nonce2_high = static_cast<uint32_t>(nonce2 >> 32);
            uint32_t nonce2_low = static_cast<uint32_t>(nonce2 & 0xFFFFFFFF);
            W2_6 = pre_swapped_base[6] ^ nonce2_high;
            W2_7 = pre_swapped_base[7] ^ nonce2_low;

            // Apply padding to both
            W1_8 = 0x80000000; W2_8 = 0x80000000;
            W1_9 = 0; W2_9 = 0;
            W1_10 = 0; W2_10 = 0;
            W1_11 = 0; W2_11 = 0;
            W1_12 = 0; W2_12 = 0;
            W1_13 = 0; W2_13 = 0;
            W1_14 = 0; W2_14 = 0;
            W1_15 = 0x00000100; W2_15 = 0x00000100;

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
            COMPUTE_W_DUAL(16); SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
            COMPUTE_W_DUAL(17); SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
            COMPUTE_W_DUAL(18); SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
            COMPUTE_W_DUAL(19); SHA1_ROUND_0_19_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);

            // Rounds 20-39
            COMPUTE_W_DUAL(20); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
            COMPUTE_W_DUAL(21); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
            COMPUTE_W_DUAL(22); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
            COMPUTE_W_DUAL(23); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
            COMPUTE_W_DUAL(24); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
            COMPUTE_W_DUAL(25); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
            COMPUTE_W_DUAL(26); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
            COMPUTE_W_DUAL(27); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
            COMPUTE_W_DUAL(28); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
            COMPUTE_W_DUAL(29); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
            COMPUTE_W_DUAL(30); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
            COMPUTE_W_DUAL(31); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);
            COMPUTE_W_DUAL(32); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
            COMPUTE_W_DUAL(33); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
            COMPUTE_W_DUAL(34); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
            COMPUTE_W_DUAL(35); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
            COMPUTE_W_DUAL(36); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
            COMPUTE_W_DUAL(37); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
            COMPUTE_W_DUAL(38); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
            COMPUTE_W_DUAL(39); SHA1_ROUND_20_39_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);

            // Rounds 40-59
            COMPUTE_W_DUAL(40); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
            COMPUTE_W_DUAL(41); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
            COMPUTE_W_DUAL(42); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
            COMPUTE_W_DUAL(43); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
            COMPUTE_W_DUAL(44); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
            COMPUTE_W_DUAL(45); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
            COMPUTE_W_DUAL(46); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
            COMPUTE_W_DUAL(47); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);
            COMPUTE_W_DUAL(48); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
            COMPUTE_W_DUAL(49); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
            COMPUTE_W_DUAL(50); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
            COMPUTE_W_DUAL(51); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
            COMPUTE_W_DUAL(52); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
            COMPUTE_W_DUAL(53); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
            COMPUTE_W_DUAL(54); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
            COMPUTE_W_DUAL(55); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
            COMPUTE_W_DUAL(56); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
            COMPUTE_W_DUAL(57); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
            COMPUTE_W_DUAL(58); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
            COMPUTE_W_DUAL(59); SHA1_ROUND_40_59_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);

            // Rounds 60-79
            COMPUTE_W_DUAL(60); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
            COMPUTE_W_DUAL(61); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
            COMPUTE_W_DUAL(62); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
            COMPUTE_W_DUAL(63); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);
            COMPUTE_W_DUAL(64); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_0, a2, b2, c2, d2, e2, W2_0);
            COMPUTE_W_DUAL(65); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_1, a2, b2, c2, d2, e2, W2_1);
            COMPUTE_W_DUAL(66); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_2, a2, b2, c2, d2, e2, W2_2);
            COMPUTE_W_DUAL(67); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_3, a2, b2, c2, d2, e2, W2_3);
            COMPUTE_W_DUAL(68); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_4, a2, b2, c2, d2, e2, W2_4);
            COMPUTE_W_DUAL(69); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_5, a2, b2, c2, d2, e2, W2_5);
            COMPUTE_W_DUAL(70); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_6, a2, b2, c2, d2, e2, W2_6);
            COMPUTE_W_DUAL(71); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_7, a2, b2, c2, d2, e2, W2_7);
            COMPUTE_W_DUAL(72); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_8, a2, b2, c2, d2, e2, W2_8);
            COMPUTE_W_DUAL(73); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_9, a2, b2, c2, d2, e2, W2_9);
            COMPUTE_W_DUAL(74); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_10, a2, b2, c2, d2, e2, W2_10);
            COMPUTE_W_DUAL(75); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_11, a2, b2, c2, d2, e2, W2_11);
            COMPUTE_W_DUAL(76); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_12, a2, b2, c2, d2, e2, W2_12);
            COMPUTE_W_DUAL(77); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_13, a2, b2, c2, d2, e2, W2_13);
            COMPUTE_W_DUAL(78); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_14, a2, b2, c2, d2, e2, W2_14);
            COMPUTE_W_DUAL(79); SHA1_ROUND_60_79_DUAL(a1, b1, c1, d1, e1, W1_15, a2, b2, c2, d2, e2, W2_15);

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

            // Check difficulty for first hash - store in thread-local buffer
            uint32_t matching_bits1 = count_leading_zeros_160bit_intel(hash1, target);
            if (matching_bits1 >= difficulty && thread_result_count < MAX_RESULTS_PER_THREAD) {
                thread_results[thread_result_count].nonce = nonce1;
                thread_results[thread_result_count].matching_bits = matching_bits1;
                thread_results[thread_result_count].difficulty_score = matching_bits1;  // Set difficulty_score
                thread_results[thread_result_count].job_version = job_ver;
                #pragma unroll
                for (int j = 0; j < 5; j++) {
                    thread_results[thread_result_count].hash[j] = hash1[j];
                }
                thread_result_count++;
            }

            // Check difficulty for second hash
            if (i + 1 < nonces_per_thread && thread_result_count < MAX_RESULTS_PER_THREAD) {
                uint32_t matching_bits2 = count_leading_zeros_160bit_intel(hash2, target);
                if (matching_bits2 >= difficulty) {
                    thread_results[thread_result_count].nonce = nonce2;
                    thread_results[thread_result_count].matching_bits = matching_bits2;
                    thread_results[thread_result_count].difficulty_score = matching_bits2;  // Set difficulty_score
                    thread_results[thread_result_count].job_version = job_ver;
                    #pragma unroll
                    for (int j = 0; j < 5; j++) {
                        thread_results[thread_result_count].hash[j] = hash2[j];
                    }
                    thread_result_count++;
                }
            }
        }

        // Store thread results in temporary buffer - no race conditions!
        if (thread_result_count > 0) {
            uint32_t base_idx = tid * MAX_RESULTS_PER_THREAD;
            for (uint32_t i = 0; i < thread_result_count; i++) {
                temp_results[base_idx + i] = thread_results[i];
            }
            thread_counts[tid] = thread_result_count;
        }
        });
    });

    // Wait for main kernel to complete
    kernel_event.wait();

    // Second phase: compact all thread results into final buffer atomically
    auto compact_event = q.submit([=](handler& h) {
        // Create stream for logging inside the kernel
        sycl::stream out(1024 * 1024, 256, h);  // Large buffer for detailed logging

        h.single_task([=]() {
            uint32_t total_results = 0;

            // Count total results across all threads
            for (int i = 0; i < total_threads; i++) {
                total_results += thread_counts[i];
            }

            // Limit to result buffer capacity
            uint32_t original_total = total_results;
            total_results = sycl::min(total_results, result_capacity);

            // Copy results to final buffer in thread order (deterministic)
            uint32_t write_idx = 0;

            for (int tid = 0; tid < total_threads && write_idx < total_results; tid++) {
                if (thread_counts[tid] > 0) {
                    uint32_t base_idx = tid * MAX_RESULTS_PER_THREAD;
                    uint32_t count = sycl::min(thread_counts[tid], total_results - write_idx);

                    for (uint32_t i = 0; i < count && write_idx < result_capacity; i++) {
                        results[write_idx] = temp_results[base_idx + i];

                        // Log nonce (split into high and low parts for formatting)
                        uint32_t nonce_high = static_cast<uint32_t>(results[write_idx].nonce >> 32);
                        uint32_t nonce_low = static_cast<uint32_t>(results[write_idx].nonce & 0xFFFFFFFF);

                        write_idx++;
                    }
                }
            }

            // Update final count atomically
            *result_count = write_idx;
        });
    });

    // Wait for compaction to complete
    compact_event.wait();

    // Free temporary buffers
    free(temp_results, q);
    free(thread_counts, q);

    return compact_event;
}

// CPU-side byte swap function for initialization
inline uint32_t bswap32_cpu(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    #include <stdlib.h>
    return _byteswap_ulong(x);
#else
    return ((x & 0xFF000000) >> 24) | ((x & 0x00FF0000) >> 8) |
           ((x & 0x0000FF00) << 8) | ((x & 0x000000FF) << 24);
#endif
}

// Initialize SYCL runtime for Intel GPU
extern "C" bool initialize_sycl_runtime() {
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

                    // Check for Intel GPU
                    if (vendor.find("Intel") != std::string::npos ||
                        name.find("Intel") != std::string::npos ||
                        name.find("Arc") != std::string::npos ||
                        name.find("Iris") != std::string::npos ||
                        name.find("UHD") != std::string::npos ||
                        name.find("Xe") != std::string::npos) {
                        selected_device = device;
                        found_intel_gpu = true;
                        break;
                    }
                }
            }
            if (found_intel_gpu) break;
        }

        if (!found_intel_gpu) {
            auto devices = device::get_devices(info::device_type::gpu);
            if (!devices.empty()) {
                selected_device = devices[0];
            } else {
                printf("No GPU devices found\n");
                return false;
            }
        }

        // Create SYCL context and queue with in-order property for better performance
        g_intel_device = new device(selected_device);
        g_sycl_context = new context(*g_intel_device);

        // Use in-order queue for better performance on Intel GPUs
        property_list props{property::queue::in_order()};
        g_sycl_queue = new queue(*g_sycl_context, *g_intel_device, props);

        // Allocate constant memory using USM
        d_base_message_sycl = malloc_device<uint32_t>(8, *g_sycl_queue);
        d_pre_swapped_base_sycl = malloc_device<uint32_t>(8, *g_sycl_queue);
        d_target_hash_sycl = malloc_device<uint32_t>(5, *g_sycl_queue);

        if (!d_base_message_sycl || !d_pre_swapped_base_sycl || !d_target_hash_sycl) {
            printf("Failed to allocate device memory for constant data\n");
            return false;
        }

        // Print device capabilities
        auto max_work_group_size = selected_device.get_info<info::device::max_work_group_size>();
        auto max_compute_units = selected_device.get_info<info::device::max_compute_units>();
        printf("Device capabilities: Max work group size: %zu, Compute units: %u\n",
               max_work_group_size, max_compute_units);

        printf("SYCL runtime initialized successfully for Intel GPU\n");
        return true;

    } catch (const sycl::exception& e) {
        printf("SYCL exception during initialization: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        printf("Standard exception during SYCL initialization: %s\n", e.what());
        return false;
    }
}

// Update base message for SYCL
extern "C" void update_base_message_sycl(const uint32_t *base_msg_words) {
    if (!g_sycl_queue || !d_base_message_sycl || !d_pre_swapped_base_sycl) {
        printf("SYCL not initialized\n");
        return;
    }

    try {
        // Copy base message to device
        g_sycl_queue->memcpy(d_base_message_sycl, base_msg_words, 8 * sizeof(uint32_t)).wait();

        // Prepare pre-swapped version
        uint32_t pre_swapped[8];
        for (int j = 0; j < 8; j++) {
            pre_swapped[j] = bswap32_cpu(base_msg_words[j]);
        }
        g_sycl_queue->memcpy(d_pre_swapped_base_sycl, pre_swapped, 8 * sizeof(uint32_t)).wait();

    } catch (const sycl::exception& e) {
        printf("SYCL exception in update_base_message_sycl: %s\n", e.what());
    }
}

// Update target hash for SYCL - CRITICAL for job updates!
extern "C" void update_target_hash_sycl(const uint32_t *target_hash) {
    if (!g_sycl_queue || !d_target_hash_sycl) {
        printf("SYCL not initialized for target hash update\n");
        return;
    }

    try {
        // Copy target hash to global device memory
        g_sycl_queue->memcpy(d_target_hash_sycl, target_hash, 5 * sizeof(uint32_t)).wait();

        printf("SYCL: Updated target hash: %08x %08x %08x %08x %08x\n",
               target_hash[0], target_hash[1], target_hash[2], target_hash[3], target_hash[4]);

    } catch (const sycl::exception& e) {
        printf("SYCL exception in update_target_hash_sycl: %s\n", e.what());
    }
}

// COMPLETE Intel job update function - updates ALL job parameters!
extern "C" void update_complete_job_sycl(const uint32_t *base_msg_words, const uint32_t *target_hash, uint64_t job_version) {
    if (!g_sycl_queue || !d_base_message_sycl || !d_pre_swapped_base_sycl || !d_target_hash_sycl) {
        return;
    }

    try {
        g_sycl_queue->memcpy(d_base_message_sycl, base_msg_words, 8 * sizeof(uint32_t)).wait();

        uint32_t pre_swapped[8];
        for (int j = 0; j < 8; j++) {
            pre_swapped[j] = bswap32_cpu(base_msg_words[j]);
        }

        g_sycl_queue->memcpy(d_pre_swapped_base_sycl, pre_swapped, 8 * sizeof(uint32_t)).wait();
        g_sycl_queue->memcpy(d_target_hash_sycl, target_hash, 5 * sizeof(uint32_t)).wait();

    } catch (const sycl::exception& e) {
        printf("SYCL exception in update_complete_job_sycl: %s\n", e.what());
    }
}

// Launch the Intel GPU mining kernel
extern "C" void launch_mining_kernel_intel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
) {
    if (!g_sycl_queue) {
        printf("SYCL queue not initialized\n");
        return;
    }

    try {
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

        int total_threads = config.blocks * config.threads_per_block;

        // Reset result count
        //g_sycl_queue->memset(pool.count, 0, sizeof(uint32_t)).wait();

        // Copy target_hash to device memory (device-to-device copy)
        //auto copy_event = g_sycl_queue->memcpy(d_target_hash_sycl, device_job.target_hash, 5 * sizeof(uint32_t));
        //copy_event.wait();

        // Now copy back from device to verify what we actually have
        //uint32_t host_target[5];
        //g_sycl_queue->memcpy(host_target, d_target_hash_sycl, 5 * sizeof(uint32_t)).wait();

        // Launch kernel with all parameters including job_version
        sycl::event kernel_event = sha1_mining_kernel_intel(
            *g_sycl_queue,
            d_target_hash_sycl,
            d_pre_swapped_base_sycl,
            difficulty,
            pool.results,
            pool.count,
            pool.capacity,
            nonce_offset,
            NONCES_PER_THREAD,
            job_version,
            total_threads
        );

        // Wait for kernel completion
        kernel_event.wait();
    } catch (const sycl::exception& e) {
        printf("SYCL exception in kernel launch: %s\n", e.what());
    } catch (const std::exception& e) {
        printf("Standard exception in kernel launch: %s\n", e.what());
    }
}

// Cleanup SYCL resources
extern "C" void cleanup_sycl_runtime() {
    if (g_sycl_queue) {
        try {
            g_sycl_queue->wait();

            if (d_base_message_sycl) {
                free(d_base_message_sycl, *g_sycl_queue);
                d_base_message_sycl = nullptr;
            }
            if (d_pre_swapped_base_sycl) {
                free(d_pre_swapped_base_sycl, *g_sycl_queue);
                d_pre_swapped_base_sycl = nullptr;
            }
            if (d_target_hash_sycl) {
                free(d_target_hash_sycl, *g_sycl_queue);
                d_target_hash_sycl = nullptr;
            }
        } catch (const sycl::exception& e) {
            printf("SYCL exception during cleanup: %s\n", e.what());
        }
    }

    delete g_sycl_queue;
    delete g_sycl_context;
    delete g_intel_device;

    g_sycl_queue = nullptr;
    g_sycl_context = nullptr;
    g_intel_device = nullptr;
}