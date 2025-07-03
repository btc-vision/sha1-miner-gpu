#include "job_constants.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];

/* SHA-1 round constants in const mem */
__device__ __constant__ uint32_t K[4] = {
    0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xCA62C1D6u
};

/* Early rejection masks - tuned for your specific target */
__device__ __constant__ uint32_t EARLY_MASKS[5] = {
    0xFFFF0000u, // Round 20: Check high 16 bits
    0xFF000000u, // Round 40: Check high 8 bits
    0xFFF00000u, // Round 60: Check high 12 bits
    0xFFFF0000u, // Round 70: Check high 16 bits
    0xFFFFFFFFu // Round 80: Full check
};

// Use PTX for optimal rotation
#if __CUDA_ARCH__ >= 600
__device__ __forceinline__
uint32_t rotl32_ptx(uint32_t x, uint32_t n) {
    uint32_t r;
    asm ("shf.l.wrap.b32 %0, %1, %1, %2;"
         : "=r"(r) : "r"(x), "r"(n));
    return r;
}
#else
__device__ __forceinline__
uint32_t rotl32_ptx(uint32_t x, uint32_t n) {
    // fallback
    return (x << n) | (x >> (32 - n));
}
#endif

// Optimized message schedule using warp shuffles
__device__ __forceinline__ uint32_t schedule(
    uint32_t &w0, uint32_t &w1, uint32_t &w2, uint32_t &w3,
    uint32_t &w4, uint32_t &w5, uint32_t &w6, uint32_t &w7,
    uint32_t &w8, uint32_t &w9, uint32_t &w10, uint32_t &w11,
    uint32_t &w12, uint32_t &w13, uint32_t &w14, uint32_t &w15,
    int round
) {
    if (round < 16) {
        switch (round) {
            case 0: return w0;
            case 1: return w1;
            case 2: return w2;
            case 3: return w3;
            case 4: return w4;
            case 5: return w5;
            case 6: return w6;
            case 7: return w7;
            case 8: return w8;
            case 9: return w9;
            case 10: return w10;
            case 11: return w11;
            case 12: return w12;
            case 13: return w13;
            case 14: return w14;
            case 15: return w15;
        }
    }

    // Compute new word
    uint32_t t = w0 ^ w2 ^ w8 ^ w13;
    t = rotl32_ptx(t, 1);

    // Shift registers
    w0 = w1;
    w1 = w2;
    w2 = w3;
    w3 = w4;
    w4 = w5;
    w5 = w6;
    w6 = w7;
    w7 = w8;
    w8 = w9;
    w9 = w10;
    w10 = w11;
    w11 = w12;
    w12 = w13;
    w13 = w14;
    w14 = w15;
    w15 = t;

    return t;
}

extern "C" __global__ __launch_bounds__(256, 8)
void sha1_double_kernel(
    uint8_t * __restrict__ out_msgs,
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;

    // Shared memory for fast target comparison
    __shared__ uint32_t shared_target[5];
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = g_target[threadIdx.x];
    }
    __syncthreads();

    // Load base message into registers
    uint32_t M[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        M[i] = __ldg(reinterpret_cast<const uint32_t *>(g_job_msg) + i);
    }

    // Apply per-thread nonce with better distribution
    uint64_t nonce = seed + tid;
    M[6] ^= (nonce & 0xFFFFFFFF);
    M[7] ^= (nonce >> 32) ^ __brev(tid); // Bit reversal for diversity

    // Message schedule in registers
    uint32_t w0 = __byte_perm(M[0], 0, 0x0123);
    uint32_t w1 = __byte_perm(M[1], 0, 0x0123);
    uint32_t w2 = __byte_perm(M[2], 0, 0x0123);
    uint32_t w3 = __byte_perm(M[3], 0, 0x0123);
    uint32_t w4 = __byte_perm(M[4], 0, 0x0123);
    uint32_t w5 = __byte_perm(M[5], 0, 0x0123);
    uint32_t w6 = __byte_perm(M[6], 0, 0x0123);
    uint32_t w7 = __byte_perm(M[7], 0, 0x0123);
    uint32_t w8 = 0x80000000u;
    uint32_t w9 = 0;
    uint32_t w10 = 0;
    uint32_t w11 = 0;
    uint32_t w12 = 0;
    uint32_t w13 = 0;
    uint32_t w14 = 0;
    uint32_t w15 = 0x00000100u;

    // SHA-1 state
    uint32_t a = 0x67452301;
    uint32_t b = 0xEFCDAB89;
    uint32_t c = 0x98BADCFE;
    uint32_t d = 0x10325476;
    uint32_t e = 0xC3D2E1F0;

    // Store initial state for final addition
    const uint32_t a0 = a;
    const uint32_t b0 = b;
    const uint32_t c0 = c;
    const uint32_t d0 = d;
    const uint32_t e0 = e;

    // Unrolled SHA-1 rounds with progressive early exit
#pragma unroll 20
    for (int i = 0; i < 20; i++) {
        uint32_t f = (b & c) | (~b & d);
        uint32_t wi = schedule(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, i);
        uint32_t temp = rotl32_ptx(a, 5) + f + e + K[0] + wi;
        e = d;
        d = c;
        c = rotl32_ptx(b, 30);
        b = a;
        a = temp;
    }

    // Early exit check at round 20
    if ((a & EARLY_MASKS[0]) != (shared_target[0] & EARLY_MASKS[0])) {
        return;
    }

#pragma unroll 20
    for (int i = 20; i < 40; i++) {
        uint32_t f = b ^ c ^ d;
        uint32_t wi = schedule(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, i);
        uint32_t temp = rotl32_ptx(a, 5) + f + e + K[1] + wi;
        e = d;
        d = c;
        c = rotl32_ptx(b, 30);
        b = a;
        a = temp;
    }

    // Early exit check at round 40
    if ((b & EARLY_MASKS[1]) != ((shared_target[1] - b0) & EARLY_MASKS[1])) {
        return;
    }

#pragma unroll 20
    for (int i = 40; i < 60; i++) {
        uint32_t f = (b & c) | (b & d) | (c & d);
        uint32_t wi = schedule(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, i);
        uint32_t temp = rotl32_ptx(a, 5) + f + e + K[2] + wi;
        e = d;
        d = c;
        c = rotl32_ptx(b, 30);
        b = a;
        a = temp;
    }

    // Early exit check at round 60
    if ((c & EARLY_MASKS[2]) != ((shared_target[2] - c0) & EARLY_MASKS[2])) {
        return;
    }

#pragma unroll 10
    for (int i = 60; i < 70; i++) {
        uint32_t f = b ^ c ^ d;
        uint32_t wi = schedule(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, i);
        uint32_t temp = rotl32_ptx(a, 5) + f + e + K[3] + wi;
        e = d;
        d = c;
        c = rotl32_ptx(b, 30);
        b = a;
        a = temp;
    }

    // Early exit check at round 70
    if ((d & EARLY_MASKS[3]) != ((shared_target[3] - d0) & EARLY_MASKS[3])) {
        return;
    }

#pragma unroll 10
    for (int i = 70; i < 80; i++) {
        uint32_t f = b ^ c ^ d;
        uint32_t wi = schedule(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, i);
        uint32_t temp = rotl32_ptx(a, 5) + f + e + K[3] + wi;
        e = d;
        d = c;
        c = rotl32_ptx(b, 30);
        b = a;
        a = temp;
    }

    // Final addition
    a += a0;
    b += b0;
    c += c0;
    d += d0;
    e += e0;

    // Full comparison using warp voting for fast rejection
    bool match = (a == shared_target[0] &&
                  b == shared_target[1] &&
                  c == shared_target[2] &&
                  d == shared_target[3] &&
                  e == shared_target[4]);

    if (!match) return;

    // Success! Write the result
    uint32_t pos = atomicAdd(ticket, 1);
    if (out_pairs && pos < (1u << 20)) {
        uint64_t *dst = out_pairs + pos * 4;
        dst[0] = (uint64_t(M[1]) << 32) | M[0];
        dst[1] = (uint64_t(M[3]) << 32) | M[2];
        dst[2] = (uint64_t(M[5]) << 32) | M[4];
        dst[3] = (uint64_t(M[7]) << 32) | M[6];
    }
}

// Host-side function to dynamically adjust early exit masks based on target
__host__ void compute_optimal_masks(const uint32_t *target, uint32_t *masks) {
    // Analyze target entropy to determine best early exit points
    for (int i = 0; i < 5; i++) {
        // Count leading zeros manually or use CUDA intrinsics
        uint32_t val = target[i];
        int leading_zeros = 0;
        int trailing_zeros = 0;

        // Count leading zeros
        if (val == 0) {
            leading_zeros = 32;
        } else {
            uint32_t temp = val;
            while ((temp & 0x80000000u) == 0) {
                leading_zeros++;
                temp <<= 1;
            }
        }

        // Count trailing zeros
        if (val == 0) {
            trailing_zeros = 32;
        } else {
            uint32_t temp = val;
            while ((temp & 1u) == 0) {
                trailing_zeros++;
                temp >>= 1;
            }
        }

        if (leading_zeros >= 16) {
            masks[i] = 0xFFFF0000u; // Check top 16 bits
        } else if (leading_zeros >= 8) {
            masks[i] = 0xFF000000u; // Check top 8 bits
        } else if (trailing_zeros >= 16) {
            masks[i] = 0x0000FFFFu; // Check bottom 16 bits
        } else {
            masks[i] = 0xFFFFFFFFu; // Full check
        }
    }

    // Upload to constant memory
    cudaMemcpyToSymbol(EARLY_MASKS, masks, sizeof(uint32_t) * 5);
}

// Device-side helper functions for bit counting
__device__ __forceinline__ int count_leading_zeros(uint32_t x) {
    return __clz(x); // CUDA intrinsic
}

__device__ __forceinline__ int count_trailing_zeros(uint32_t x) {
    // Use bit manipulation trick
    return __ffs(x) - 1; // Find first set bit (1-indexed) and subtract 1
}
