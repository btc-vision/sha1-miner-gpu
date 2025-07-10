#include <iostream>

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
 * Optimized for RDNA4 with dynamic work distribution
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

    uint8_t base_msg[32];
    uint4* base_msg_vec = (uint4*)base_msg;
    const uint4* base_message_vec = (const uint4*)base_message;
    base_msg_vec[0] = base_message_vec[0];  // Loads 16 bytes
    base_msg_vec[1] = base_message_vec[1];  // Loads next 16 bytes

    // Load target (already in correct format)
    uint32_t target[5];
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = target_hash[i];
    }

    // Track processed nonces
    uint32_t nonces_processed = 0;

    // Main mining loop
    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0) continue;

        nonces_processed++;

        // Create a copy of the message
        uint8_t msg_bytes[32];

        // Vectorized version - 2 operations
        uint4* msg_bytes_vec = (uint4*)msg_bytes;
        msg_bytes_vec[0] = base_msg_vec[0];  // Copies bytes 0-15
        msg_bytes_vec[1] = base_msg_vec[1];  // Copies bytes 16-31

        // Apply nonce to last 8 bytes by XORing (big-endian)
        uint32_t* msg_words = (uint32_t*)msg_bytes;
        msg_words[6] ^= __builtin_bswap32((uint32_t)(nonce >> 32));
        msg_words[7] ^= __builtin_bswap32((uint32_t)(nonce & 0xFFFFFFFF));

        // Convert message bytes to big-endian words for SHA-1 - MATCHING NVIDIA
        uint32_t W[16];

        uint32_t* msg_words_local = (uint32_t*)msg_bytes;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = __builtin_bswap32(msg_words_local[j]);
        }

        // Apply SHA-1 padding
        W[8] = 0x80000000; // Padding bit
#pragma unroll
        for (int j = 9; j < 15; j++) {
            W[j] = 0;
        }
        W[15] = 0x00000100; // Message length: 256 bits in big-endian

        // Initialize hash values
        uint32_t a = H0[0];
        uint32_t b = H0[1];
        uint32_t c = H0[2];
        uint32_t d = H0[3];
        uint32_t e = H0[4];

        // SHA-1 rounds 0-19
#pragma unroll
        for (int t = 0; t < 20; t++) {
            if (t >= 16) {
                uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                W[(t - 14) & 15] ^ W[(t - 16) & 15];
                W[t & 15] = amd_rotl32(temp, 1);
            }
            uint32_t f = (b & c) | (~b & d);
            uint32_t temp = amd_rotl32(a, 5) + f + e + K[0] + W[t & 15];
            e = d;
            d = c;
            c = amd_rotl32(b, 30);
            b = a;
            a = temp;
        }

        // Rounds 20-39
#pragma unroll
        for (int t = 20; t < 40; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                            W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = amd_rotl32(temp, 1);
            uint32_t f = b ^ c ^ d;
            uint32_t temp2 = amd_rotl32(a, 5) + f + e + K[1] + W[t & 15];
            e = d;
            d = c;
            c = amd_rotl32(b, 30);
            b = a;
            a = temp2;
        }

        // Rounds 40-59
#pragma unroll
        for (int t = 40; t < 60; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                            W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = amd_rotl32(temp, 1);
            uint32_t f = (b & c) | (b & d) | (c & d);
            uint32_t temp2 = amd_rotl32(a, 5) + f + e + K[2] + W[t & 15];
            e = d;
            d = c;
            c = amd_rotl32(b, 30);
            b = a;
            a = temp2;
        }

        // Rounds 60-79
#pragma unroll
        for (int t = 60; t < 80; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                            W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = amd_rotl32(temp, 1);
            uint32_t f = b ^ c ^ d;
            uint32_t temp2 = amd_rotl32(a, 5) + f + e + K[3] + W[t & 15];
            e = d;
            d = c;
            c = amd_rotl32(b, 30);
            b = a;
            a = temp2;
        }

        // Add initial hash values
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
 * Properly configured for RDNA4 and other AMD architectures
 */
extern "C" void launch_mining_kernel_amd(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
) {
    // Get device properties once and cache
    static thread_local hipDeviceProp_t props_cached;
    static thread_local bool props_initialized = false;
    static thread_local int last_device_id = -1;

    int current_device;
    hipGetDevice(&current_device);

    if (!props_initialized || current_device != last_device_id) {
        hipError_t err = hipGetDeviceProperties(&props_cached, current_device);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to get device properties: %s\n", hipGetErrorString(err));
            return;
        }
        props_initialized = true;
        last_device_id = current_device;
    }

    // Use cached properties
    hipDeviceProp_t &props = props_cached;

    // Optimize configuration for AMD
    int blocks = config.blocks;
    int threads = config.threads_per_block;

    // AMD-specific optimizations
    if (blocks <= 0) {
        // Calculate optimal blocks based on architecture
        int compute_units = props.multiProcessorCount;
        int waves_per_cu = 4; // Reduced for better scheduling
        int total_waves = compute_units * waves_per_cu;
        blocks = (total_waves * props.warpSize) / threads;
    }

    // Dynamic nonces_per_thread calculation for proper hash counting
    uint32_t nonces_per_thread;
    //uint32_t target_nonces_per_kernel;

    // Detect architecture from device properties
    //std::string arch_name = props.gcnArchName ? props.gcnArchName : "";
    //std::string device_name = props.name ? props.name : "";

    /*bool is_rdna4 = (arch_name.find("gfx12") != std::string::npos) ||
                    (props.major == 12) ||
                    (device_name.find("9070") != std::string::npos) ||
                    (device_name.find("9080") != std::string::npos);
    bool is_rdna3 = (arch_name.find("gfx11") != std::string::npos) ||
                    (props.major == 11) ||
                    (device_name.find("7900") != std::string::npos) ||
                    (device_name.find("7800") != std::string::npos) ||
                    (device_name.find("7700") != std::string::npos);
    bool is_rdna2 = (arch_name.find("gfx103") != std::string::npos) ||
                    (device_name.find("6900") != std::string::npos) ||
                    (device_name.find("6800") != std::string::npos) ||
                    (device_name.find("6700") != std::string::npos);
    bool is_rdna1 = (arch_name.find("gfx101") != std::string::npos) ||
                (arch_name.find("gfx1010") != std::string::npos) ||
                (device_name.find("5700") != std::string::npos) ||
                (device_name.find("5600") != std::string::npos) ||
                (device_name.find("5500") != std::string::npos);

    if (is_rdna4) {
        target_nonces_per_kernel = NONCES_PER_THREAD_RDNA4;
    } else if (is_rdna3) {
        target_nonces_per_kernel = NONCES_PER_THREAD_RDNA3;
    } else if (is_rdna2) {
        target_nonces_per_kernel = NONCES_PER_THREAD_RDNA2;
    } else if (is_rdna1) {
        target_nonces_per_kernel = NONCES_PER_THREAD_RDNA1;
    } else {
        target_nonces_per_kernel = NONCES_PER_THREAD; // Default
    }*/

    // Calculate nonces per thread
    uint64_t total_threads = static_cast<uint64_t>(blocks) * static_cast<uint64_t>(threads);
    nonces_per_thread = NONCES_PER_THREAD / total_threads;

    // Ensure minimum work per thread based on architecture
    /*uint32_t min_nonces;
    if (is_rdna4) {
        min_nonces = 64;  // RDNA4 minimum
    } else if (is_rdna3) {
        min_nonces = 64;  // RDNA3 minimum
    } else if (is_rdna2) {
        min_nonces = 56;  // RDNA2 minimum
    } else if (is_rdna1) {
        min_nonces = 56;  // RDNA1 needs more work per thread
    } else {
        min_nonces = 48;  // Older GPUs
    }

    // Ensure we have at least minimum work per thread
    if (nonces_per_thread < min_nonces) {
        nonces_per_thread = min_nonces;
    }*/

    // For very high thread counts, ensure we don't overflow
    if (nonces_per_thread == 0) {
        nonces_per_thread = 1;
    }

    // Reset result count asynchronously
    hipError_t err = hipMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to reset result count: %s\n", hipGetErrorString(err));
        return;
    }

    // Launch configuration
    dim3 gridDim(blocks);
    dim3 blockDim(threads);
    size_t lds_size = 0;

    // Launch kernel with calculated work per thread
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

    // Check for launch errors
    //err = hipGetLastError();
    //if (err != hipSuccess) {
    //    fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(err));
    //}
}
