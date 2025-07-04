#include "job_constants.cu"
#include <cuda_runtime.h>
#include <cstdio>

// SHA-1 constants
#define K0 0x5A827999u
#define K1 0x6ED9EBA1u
#define K2 0x8F1BBCDCu
#define K3 0xCA62C1D6u

#define H0 0x67452301u
#define H1 0xEFCDAB89u
#define H2 0x98BADCFEu
#define H3 0x10325476u
#define H4 0xC3D2E1F0u

// Differential path constants
#define NEUTRAL_BITS_MASK 0x0000FFFFu
#define DISTURBANCE_VECTOR 0x80000000u

// Use global memory instead of constant memory for writable data
__device__ uint32_t g_differential_path[80];
__device__ uint32_t g_message_conditions[16];
__device__ uint32_t g_near_collision_blocks[2][16];

// Helper functions
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ uint32_t swap32(uint32_t x) {
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) |
           ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24);
}

// SHA-1 round functions
__device__ __forceinline__ uint32_t f1(uint32_t b, uint32_t c, uint32_t d) {
    return (b & c) | (~b & d);
}

__device__ __forceinline__ uint32_t f2(uint32_t b, uint32_t c, uint32_t d) {
    return b ^ c ^ d;
}

__device__ __forceinline__ uint32_t f3(uint32_t b, uint32_t c, uint32_t d) {
    return (b & c) | (b & d) | (c & d);
}

// Message expansion with differential modification
__device__ void expand_message_differential(uint32_t W[80], uint32_t differential_mask) {
    for (int t = 16; t < 80; t++) {
        W[t] = W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16];
        W[t] = rotl32(W[t], 1);

        if (g_differential_path[t] != 0) {
            W[t] ^= (g_differential_path[t] & differential_mask);
        }
    }
}

// Compute SHA-1 with differential path constraints
__device__ void sha1_differential(const uint32_t W[80], uint32_t state[5],
                                  uint32_t path_conditions[80]) {
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];

    bool on_path = true;

    for (int t = 0; t < 80 && on_path; t++) {
        uint32_t f, k;

        if (t < 20) {
            f = f1(b, c, d);
            k = K0;
        } else if (t < 40) {
            f = f2(b, c, d);
            k = K1;
        } else if (t < 60) {
            f = f3(b, c, d);
            k = K2;
        } else {
            f = f2(b, c, d);
            k = K3;
        }

        uint32_t temp = rotl32(a, 5) + f + e + k + W[t];

        if (path_conditions[t] != 0) {
            uint32_t condition_mask = path_conditions[t];
            if ((temp & condition_mask) != (g_differential_path[t] & condition_mask)) {
                on_path = false;
            }
        }

        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    if (on_path) {
        state[0] = a + state[0];
        state[1] = b + state[1];
        state[2] = c + state[2];
        state[3] = d + state[3];
        state[4] = e + state[4];
    } else {
        state[0] |= 0x80000000u;
    }
}

// Near-collision block construction
__device__ void construct_near_collision_block(uint32_t block[16],
                                               uint32_t prefix_hash[5],
                                               uint32_t neutral_bits,
                                               int block_type) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
        block[i] = g_near_collision_blocks[block_type][i];
    }

    block[0] ^= (neutral_bits & 0x0000FFFF);
    block[1] ^= ((neutral_bits >> 16) & 0x0000FFFF);

#pragma unroll
    for (int i = 0; i < 5; i++) {
        if (g_message_conditions[i] != 0) {
            block[i] = (block[i] & ~g_message_conditions[i]) |
                       (prefix_hash[i] & g_message_conditions[i]);
        }
    }
}

// Boomerang attack for connecting near-collision blocks
__device__ bool boomerang_connect(uint32_t block1[16], uint32_t block2[16],
                                  uint32_t intermediate_hash[5]) {
    uint32_t state1[5];

#pragma unroll
    for (int i = 0; i < 5; i++) {
        state1[i] = intermediate_hash[i];
    }

    uint32_t W1[80];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        W1[i] = swap32(block1[i]);
    }
    expand_message_differential(W1, DISTURBANCE_VECTOR);

    uint32_t path_conditions[80] = {0};
    sha1_differential(W1, state1, path_conditions);

    bool connected = true;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        uint32_t diff = state1[i] ^ intermediate_hash[i];
        if (__popc(diff) > 5) {
            connected = false;
        }
    }

    return connected;
}

// Initialize constants - now properly writes to global memory
__global__ void init_shattered_constants_kernel() {
    if (threadIdx.x >= 20 && threadIdx.x < 40) {
        g_differential_path[threadIdx.x] = 0x80000000u >> (threadIdx.x - 20);
    }

    if (threadIdx.x < 16) {
        g_message_conditions[threadIdx.x] = (threadIdx.x < 4) ? 0xFFFF0000u : 0;
    }

    if (threadIdx.x < 16) {
        g_near_collision_blocks[0][threadIdx.x] = 0x12345678u ^ (threadIdx.x << 24);
        g_near_collision_blocks[1][threadIdx.x] = 0x87654321u ^ (threadIdx.x << 16);
    }
}

extern "C" void init_shattered_constants() {
    // Launch the kernel
    init_shattered_constants_kernel<<<1, 64>>>();
    cudaDeviceSynchronize();
}

// Main SHAttered-style collision kernel
extern "C" __global__ __launch_bounds__(256, 4)
void sha1_shattered_collision_kernel(
    uint64_t * __restrict__ out_pairs,
    uint32_t * __restrict__ ticket,
    uint64_t seed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    //const uint32_t lane_id = threadIdx.x & 31;
    //const uint32_t warp_id = tid >> 5;

    // Use lane_id to suppress warning
    //(void) lane_id;

    __shared__ uint32_t s_prefix_hash[5];
    __shared__ uint32_t s_target[5];
    __shared__ uint32_t s_found_collision;

    if (threadIdx.x == 0) {
        s_found_collision = 0;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            s_target[i] = __ldg(&g_target[i]);
        }

        uint32_t prefix[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            prefix[i] = ((uint32_t *) g_job_msg)[i];
        }

        uint32_t W[16];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            W[i] = swap32(prefix[i]);
        }
        W[8] = 0x80000000;
#pragma unroll
        for (int i = 9; i < 15; i++) {
            W[i] = 0;
        }
        W[15] = 256;

        uint32_t a = H0, b = H1, c = H2, d = H3, e = H4;

#pragma unroll
        for (int t = 0; t < 20; t++) {
            uint32_t f = f1(b, c, d);
            uint32_t temp = rotl32(a, 5) + f + e + K0 + W[t & 15];
            e = d;
            d = c;
            c = rotl32(b, 30);
            b = a;
            a = temp;
        }

        s_prefix_hash[0] = a + H0;
        s_prefix_hash[1] = b + H1;
        s_prefix_hash[2] = c + H2;
        s_prefix_hash[3] = d + H3;
        s_prefix_hash[4] = e + H4;
    }
    __syncthreads();

    const uint32_t TRIALS_PER_THREAD = 256;
    uint32_t base_neutral = seed + tid * TRIALS_PER_THREAD;

    for (uint32_t trial = 0; trial < TRIALS_PER_THREAD; trial++) {
        if (s_found_collision) break;

        uint32_t neutral_bits = base_neutral + trial;

        uint32_t block1[16], block2[16];
        construct_near_collision_block(block1, s_prefix_hash, neutral_bits, 0);

        uint32_t intermediate[5];
#pragma unroll
        for (int i = 0; i < 5; i++) {
            intermediate[i] = s_prefix_hash[i];
        }

        uint32_t W[80];
#pragma unroll
        for (int i = 0; i < 16; i++) {
            W[i] = swap32(block1[i]);
        }
        expand_message_differential(W, neutral_bits);

        uint32_t path_conditions[80] = {0};
        sha1_differential(W, intermediate, path_conditions);

        if (intermediate[0] & 0x80000000u) continue;

        for (uint32_t nb2 = 0; nb2 < 16; nb2++) {
            construct_near_collision_block(block2, intermediate,
                                           (neutral_bits << 4) | nb2, 1);

            if (boomerang_connect(block1, block2, intermediate)) {
                uint32_t final_state[5];
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    final_state[i] = intermediate[i];
                }

#pragma unroll
                for (int i = 0; i < 16; i++) {
                    W[i] = swap32(block2[i]);
                }
                expand_message_differential(W, (neutral_bits << 4) | nb2);
                sha1_differential(W, final_state, path_conditions);

                bool match = true;
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    if (final_state[i] != s_target[i]) {
                        match = false;
                        break;
                    }
                }

                if (match && atomicCAS(&s_found_collision, 0, 1) == 0) {
                    uint32_t pos = atomicAdd(ticket, 1);
                    if (pos < (1u << 20)) {
                        uint64_t *dst = out_pairs + pos * 16;

#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            uint32_t lo = block1[i * 2];
                            uint32_t hi = block1[i * 2 + 1];
                            dst[i] = ((uint64_t) hi << 32) | lo;
                        }

#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            uint32_t lo = block2[i * 2];
                            uint32_t hi = block2[i * 2 + 1];
                            dst[8 + i] = ((uint64_t) hi << 32) | lo;
                        }
                    }
                    return;
                }
            }
        }

        if ((trial & 15) == 15) {
            __syncthreads();

            uint32_t quality = __popc(intermediate[0] ^ s_target[0]);
            uint32_t best_quality = quality;

#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                uint32_t other_quality = __shfl_down_sync(0xFFFFFFFF, quality, offset);
                if (other_quality < best_quality) {
                    best_quality = other_quality;
                }
            }

            if (quality == best_quality && quality < 8) {
                for (uint32_t birthday = 0; birthday < 64; birthday++) {
                    uint32_t special_neutral = (neutral_bits << 16) | (birthday << 8) | tid;

                    construct_near_collision_block(block2, intermediate,
                                                   special_neutral, 1);

                    uint32_t collision_state[5];
#pragma unroll
                    for (int i = 0; i < 5; i++) {
                        collision_state[i] = intermediate[i];
                    }

#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        W[i] = swap32(block2[i]) ^ ((i == 5) ? DISTURBANCE_VECTOR : 0);
                    }
                    expand_message_differential(W, special_neutral);
                    sha1_differential(W, collision_state, path_conditions);

                    bool match = true;
#pragma unroll
                    for (int i = 0; i < 5; i++) {
                        if (collision_state[i] != s_target[i]) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
                        atomicExch(&s_found_collision, 1);
                        return;
                    }
                }
            }
        }
    }
}

// Debug kernel
extern "C" __global__ void sha1_shattered_debug_kernel(
    uint32_t * __restrict__ debug_output,
    uint64_t seed
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("SHAttered-style collision attack kernel initialized\n");
        printf("Seed: %llu\n", seed);
        printf("Using differential cryptanalysis\n");

        uint32_t test_state[5] = {H0, H1, H2, H3, H4};
        uint32_t test_W[80] = {0};
        test_W[0] = 0x61626364u;

        uint32_t path_conditions[80] = {0};
        sha1_differential(test_W, test_state, path_conditions);

        printf("Test differential SHA-1: %08x %08x %08x %08x %08x\n",
               test_state[0], test_state[1], test_state[2],
               test_state[3], test_state[4]);

        if (debug_output) {
            debug_output[0] = 0xDEADBEEF;
            for (int i = 0; i < 5; i++) {
                debug_output[i + 1] = test_state[i];
            }
        }
    }
}
