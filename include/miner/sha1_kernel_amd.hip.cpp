// sha1_kernel_amd.hip.cpp - Highly optimized AMD HIP SHA-1 near-collision mining kernel
#include "sha1_miner.cuh"
#include "gpu_platform.hpp"

// SHA-1 constants in constant memory
__constant__ uint32_t K[4] = {
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6
};

__constant__ uint32_t H0[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

// AMD-specific optimizations
#define AMD_WAVEFRONT_SIZE 64  // For RDNA, runtime detection is better
#define LDS_BANK_CONFLICT_FREE_SIZE 33  // Avoid bank conflicts

/**
 * Count leading zero bits using AMD-specific intrinsics
 */
__device__ __forceinline__ uint32_t count_leading_zeros_160bit_amd(
    const uint32_t hash[5],
    const uint32_t target[5]
) {
    uint32_t total_bits = 0;
 // Use vector operations where possible
#pragma unroll
    for (int i = 0; i < 5; i++) {
        uint32_t xor_val = hash[i] ^ target[i];
        if (xor_val == 0) {
            total_bits += 32;
        } else {
            // AMD has native clz instruction
            total_bits += __builtin_clz(xor_val);
            break;
        }
    }
    return total_bits;
}

/**
 * SHA-1 F-functions optimized for AMD
 * Use standard operations that compiler can optimize
 */
__device__ __forceinline__ uint32_t amd_f1(uint32_t b, uint32_t c, uint32_t d) {
    // (b & c) | (~b & d)
    // This pattern is recognized by the compiler and optimized to V_BFI_B32 on GCN
    return (b & c) | (~b & d);
}

__device__ __forceinline__ uint32_t amd_f2(uint32_t b, uint32_t c, uint32_t d) {
    return b ^ c ^ d;
}

__device__ __forceinline__ uint32_t amd_f3(uint32_t b, uint32_t c, uint32_t d) {
    // (b & c) | (b & d) | (c & d) -> majority function
    // Can be optimized to: (b & c) | (d & (b ^ c))
    return (b & c) | (d & (b ^ c));
}

/**
 * AMD-optimized rotation using native rotate instruction
 */
__device__ __forceinline__ uint32_t amd_rotl32(uint32_t x, uint32_t n) {
    // Standard rotate left - compiler will optimize this
    return (x << n) | (x >> (32 - n));
}

/**
 * Optimized SHA-1 round function for AMD
 * Reduces register pressure and improves ILP
 */
__device__ __forceinline__ void sha1_round_amd(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t &e,
    uint32_t f, uint32_t k, uint32_t w
) {
    uint32_t temp = amd_rotl32(a, 5) + f + e + k + w;
    e = d;
    d = c;
    c = amd_rotl32(b, 30);
    b = a;
    a = temp;
}

/**
 * Message expansion optimized for AMD
 * Uses vector operations where possible
 */
__device__ __forceinline__ void expand_message_amd(uint32_t W[16], int t) {
    uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                    W[(t - 14) & 15] ^ W[(t - 16) & 15];
    W[t & 15] = amd_rotl32(temp, 1);
}

/**
 * Main SHA-1 mining kernel for AMD GPUs
 * Optimized for GCN/RDNA architectures
 */
__global__ void sha1_mining_kernel_amd(
    const uint8_t * __restrict__ base_message,
    const uint32_t * __restrict__ target_hash,
    uint32_t difficulty,
    MiningResult * __restrict__ results,
    uint32_t * __restrict__ result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread,
    uint64_t * __restrict__ actual_nonces_processed
) {
    // Thread indices
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & 63; // Assume 64-thread wavefront
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // LDS optimization - use bank conflict free addressing
    __shared__ uint32_t lds_message[8]; // Store as words, not bytes
    __shared__ uint32_t lds_target[5];

    // Cooperative load with coalesced access
    if (threadIdx.x < 8) {
        lds_message[threadIdx.x] = __builtin_bswap32(
            *((const uint32_t *) &base_message[threadIdx.x * 4])
        );
    }
    if (threadIdx.x < 5) {
        lds_target[threadIdx.x] = target_hash[threadIdx.x];
    }
    __syncthreads();

    // Load into registers (minimize register usage)
    uint32_t msg_words[8];
    uint32_t target[5];

#pragma unroll
    for (int i = 0; i < 8; i++) {
        msg_words[i] = lds_message[i];
    }
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = lds_target[i];
    }

    // Track processed nonces
    uint32_t nonces_processed = 0;

    // Main mining loop - optimize for AMD's VALU utilization
#pragma unroll 2  // Less aggressive unrolling for AMD
    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0) continue;

        nonces_processed++;

        // Message schedule array - keep in registers
        uint32_t W[16];

        // Copy first 6 words unchanged
#pragma unroll
        for (int j = 0; j < 6; j++) {
            W[j] = msg_words[j];
        }

        // Apply nonce to words 6 and 7
        uint64_t nonce_be = __builtin_bswap64(nonce);
        W[6] = msg_words[6] ^ (uint32_t) (nonce_be >> 32);
        W[7] = msg_words[7] ^ (uint32_t) (nonce_be & 0xFFFFFFFF);

        // SHA-1 padding
        W[8] = 0x80000000;
#pragma unroll
        for (int j = 9; j < 15; j++) {
            W[j] = 0;
        }
        W[15] = 0x00000100;

        // Initialize working variables
        uint32_t a = H0[0];
        uint32_t b = H0[1];
        uint32_t c = H0[2];
        uint32_t d = H0[3];
        uint32_t e = H0[4];

        // Rounds 0-15: Use original message words
#pragma unroll 16
        for (int t = 0; t < 16; t++) {
            sha1_round_amd(a, b, c, d, e, amd_f1(b, c, d), K[0], W[t]);
        }

        // Rounds 16-19: Message expansion begins
#pragma unroll 4
        for (int t = 16; t < 20; t++) {
            expand_message_amd(W, t);
            sha1_round_amd(a, b, c, d, e, amd_f1(b, c, d), K[0], W[t & 15]);
        }

        // Rounds 20-39
#pragma unroll 20
        for (int t = 20; t < 40; t++) {
            expand_message_amd(W, t);
            sha1_round_amd(a, b, c, d, e, amd_f2(b, c, d), K[1], W[t & 15]);
        }

        // Rounds 40-59
#pragma unroll 20
        for (int t = 40; t < 60; t++) {
            expand_message_amd(W, t);
            sha1_round_amd(a, b, c, d, e, amd_f3(b, c, d), K[2], W[t & 15]);
        }

        // Rounds 60-79
#pragma unroll 20
        for (int t = 60; t < 80; t++) {
            expand_message_amd(W, t);
            sha1_round_amd(a, b, c, d, e, amd_f2(b, c, d), K[3], W[t & 15]);
        }

        // Final hash computation
        uint32_t hash[5];
        hash[0] = a + H0[0];
        hash[1] = b + H0[1];
        hash[2] = c + H0[2];
        hash[3] = d + H0[3];
        hash[4] = e + H0[4];

        // Count matching bits
        uint32_t matching_bits = count_leading_zeros_160bit_amd(hash, target);

        // Check if we found a match
        if (matching_bits >= difficulty) {
            // Use AMD's wavefront vote operations
            uint64_t mask = __ballot(matching_bits >= difficulty);

            // Use ds_swizzle for efficient cross-lane communication
            if (mask != 0) {
                // Count set bits in mask up to our lane
                uint32_t lane_mask = (1ULL << lane_id) - 1;
                uint32_t prefix_sum = __builtin_popcountll(mask & lane_mask);

                // First lane does the atomic add
                uint32_t base_idx;
                if (lane_id == __builtin_ffsll(mask) - 1) {
                    base_idx = atomicAdd(result_count, __builtin_popcountll(mask));
                }

                // Broadcast base index to all lanes
                base_idx = __shfl(base_idx, __builtin_ffsll(mask) - 1, 64);

                // Each matching lane writes its result
                if ((mask >> lane_id) & 1) {
                    uint32_t idx = base_idx + prefix_sum;
                    if (idx < result_capacity) {
                        results[idx].nonce = nonce;
                        results[idx].matching_bits = matching_bits;
                        results[idx].difficulty_score = matching_bits;
#pragma unroll
                        for (int j = 0; j < 5; j++) {
                            results[idx].hash[j] = hash[j];
                        }
                    }
                }
            }
        }
    }

    // Update processed nonce count
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long *) actual_nonces_processed,
                  (unsigned long long) (blockDim.x * nonces_processed));
    }
}

/**
 * AMD-specific kernel configuration
 */
struct AMDKernelConfig {
    int compute_units;
    int waves_per_cu;
    int threads_per_block;
    int blocks;

    static AMDKernelConfig get_optimal_config(hipDeviceProp_t &props) {
        AMDKernelConfig config;

        // Get compute unit count
        config.compute_units = props.multiProcessorCount;

        // Optimal threads per block for AMD (multiple of wavefront size)
        config.threads_per_block = 256; // 4 wavefronts

        // Target 4-8 waves per CU for good occupancy
        config.waves_per_cu = 6;

        // Calculate blocks
        int total_waves = config.compute_units * config.waves_per_cu;
        config.blocks = (total_waves * 64) / config.threads_per_block;

        return config;
    }
};

/**
 * Launch the optimized AMD HIP SHA-1 mining kernel
 */
extern "C" void launch_mining_kernel_amd(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
) {
    // Get device properties for optimal configuration
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", hipGetErrorString(err));
        return;
    }

    // Get AMD-optimized configuration
    AMDKernelConfig amd_config = AMDKernelConfig::get_optimal_config(props);

    // Use provided config but optimize if needed
    int blocks = config.blocks > 0 ? config.blocks : amd_config.blocks;
    int threads = config.threads_per_block > 0 ? config.threads_per_block : amd_config.threads_per_block;

    // AMD GPUs benefit from fewer nonces per thread due to better occupancy
    uint32_t nonces_per_thread = 32; // Reduced from default

    // Reset result count
    err = hipMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to reset result count: %s\n", hipGetErrorString(err));
        return;
    }

    // Launch configuration
    dim3 gridDim(blocks);
    dim3 blockDim(threads);

    // Calculate dynamic LDS size if needed
    size_t lds_size = 0; // We use static LDS allocation

    // Launch kernel
    hipLaunchKernelGGL(
        sha1_mining_kernel_amd,
        gridDim,
        blockDim,
        lds_size,
        config.stream,
        device_job.base_message,
        device_job.target_hash,
        difficulty,
        pool.results,
        pool.count,
        pool.capacity,
        nonce_offset,
        nonces_per_thread,
        pool.nonces_processed
    );

    // Check for errors
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        fprintf(stderr, "AMD kernel launch failed: %s\n", hipGetErrorString(launch_err));
    }
}