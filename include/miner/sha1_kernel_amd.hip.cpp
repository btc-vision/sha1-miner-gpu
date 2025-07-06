// sha1_kernel_amd.hip.cpp - AMD HIP optimized SHA-1 near-collision mining kernel
#include "sha1_miner.cuh"
#include "gpu_platform.hpp"
#include "utilities.hpp"

// SHA-1 constants in constant memory
__constant__ uint32_t K[4] = {
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6
};

__constant__ uint32_t H0[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/**
 * Count leading zero bits between hash and target (XOR distance)
 * Optimized for AMD GCN/RDNA architectures
 */
__device__ __forceinline__ uint32_t count_leading_zeros_160bit_amd(
    const uint32_t hash[5],
    const uint32_t target[5]
) {
    uint32_t total_bits = 0;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        uint32_t xor_val = hash[i] ^ target[i];
        if (xor_val == 0) {
            total_bits += 32;
        } else {
            total_bits += __clz(xor_val);
            break;
        }
    }
    return total_bits;
}

/**
 * AMD-optimized rotation using native rotate instruction
 */
__device__ __forceinline__ uint32_t amd_rotl32(uint32_t x, uint32_t n) {
    return __builtin_rotateleft32(x, n);
}

/**
 * Main SHA-1 mining kernel for AMD GPUs
 * Optimized for GCN/RDNA with 64-thread wavefronts
 */
__global__ void sha1_mining_kernel_amd(
    const uint8_t * __restrict__ base_message,
    const uint32_t * __restrict__ target_hash,
    uint32_t difficulty,
    MiningResult * __restrict__ results,
    uint32_t * __restrict__ result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread
) {
    // Thread indices - AMD uses 64-thread wavefronts
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & 63; // Lane within wavefront
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // LDS (Local Data Share) for cooperative loading
    __shared__ uint8_t lds_base_message[32];
    __shared__ uint32_t lds_target_hash[5];

    // Cooperative load of message and target using first wavefront
    if (threadIdx.x < 32) {
        lds_base_message[threadIdx.x] = base_message[threadIdx.x];
    }
    if (threadIdx.x < 5) {
        lds_target_hash[threadIdx.x] = target_hash[threadIdx.x];
    }
    __syncthreads();

    // Load into registers
    uint8_t base_msg[32];
    uint32_t target[5];
#pragma unroll
    for (int i = 0; i < 32; i++) {
        base_msg[i] = lds_base_message[i];
    }
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = lds_target_hash[i];
    }

    // Process nonces - AMD prefers less aggressive unrolling
#pragma unroll 4
    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0) continue;

        // Create message copy
        uint8_t msg_bytes[32];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            msg_bytes[j] = base_msg[j];
        }

        // Apply nonce to last 8 bytes (big-endian)
        uint64_t nonce_be = __builtin_bswap64(nonce);
        *((uint64_t *) &msg_bytes[24]) ^= nonce_be;

        // Convert message bytes to big-endian words
        uint32_t W[16];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = __builtin_bswap32(*((uint32_t *) &msg_bytes[j * 4]));
        }

        // Apply SHA-1 padding
        W[8] = 0x80000000;
#pragma unroll
        for (int j = 9; j < 15; j++) {
            W[j] = 0;
        }
        W[15] = 0x00000100;

        // Initialize hash values
        uint32_t a = H0[0];
        uint32_t b = H0[1];
        uint32_t c = H0[2];
        uint32_t d = H0[3];
        uint32_t e = H0[4];

        // SHA-1 rounds 0-19
#pragma unroll 2
        for (int t = 0; t < 20; t++) {
            if (t >= 16) {
                uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
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
#pragma unroll 2
        for (int t = 20; t < 40; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
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
#pragma unroll 2
        for (int t = 40; t < 60; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
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
#pragma unroll 2
        for (int t = 60; t < 80; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
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
            // AMD wavefront vote operations
            uint64_t mask = __ballot(matching_bits >= difficulty);
            // Only first active lane in wavefront performs atomic
            if (lane_id == __ffsll(mask) - 1) {
                uint32_t idx = atomicAdd(result_count, __popcll(mask));
                // Store results for all matching threads in wavefront
                for (int bit = __ffsll(mask) - 1; bit >= 0; bit = __ffsll(mask & ~((1ULL << (bit + 1)) - 1)) - 1) {
                    if (idx < result_capacity) {
                        uint32_t source_lane = bit;
                        // Use AMD's permute lane operations
                        uint64_t result_nonce = __shfl(nonce, source_lane, 64);
                        uint32_t result_bits = __shfl(matching_bits, source_lane, 64);

                        results[idx].nonce = result_nonce;
                        results[idx].matching_bits = result_bits;
                        results[idx].difficulty_score = result_bits;

#pragma unroll
                        for (int j = 0; j < 5; j++) {
                            results[idx].hash[j] = __shfl(hash[j], source_lane, 64);
                        }
                        idx++;
                    }
                }
            }
        }
    }
}

/**
 * Launch the AMD HIP SHA-1 mining kernel
 */
void launch_mining_kernel_amd(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
) {
    // Validate configuration
    if (config.blocks <= 0 || config.threads_per_block <= 0) {
        fprintf(stderr, "Invalid kernel configuration: blocks=%d, threads=%d\n",
                config.blocks, config.threads_per_block);
        return;
    }

    // AMD GPUs prefer fewer nonces per thread due to different memory hierarchy
    uint32_t nonces_per_thread = NONCES_PER_THREAD;

    // Reset result count
    hipError_t err = hipMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to reset result count: %s\n", hipGetErrorString(err));
        return;
    }

    // Clear previous errors
    hipGetLastError();

    // Launch configuration
    dim3 gridDim(config.blocks, 1, 1);
    dim3 blockDim(config.threads_per_block, 1, 1);

#ifdef DEBUG_SHA1
    printf("[DEBUG] Launching AMD kernel: blocks=%d, threads=%d\n",
           config.blocks, config.threads_per_block);
    printf("[DEBUG] Mining parameters: difficulty=%u, nonce_offset=%llu, nonces_per_thread=%u\n",
           difficulty, nonce_offset, nonces_per_thread);
#endif

    // Launch kernel using HIP
    hipLaunchKernelGGL(sha1_mining_kernel_amd, gridDim, blockDim, 0, config.stream,
                       device_job.base_message,
                       device_job.target_hash,
                       difficulty,
                       pool.results,
                       pool.count,
                       pool.capacity,
                       nonce_offset,
                       nonces_per_thread
    );

    // Check for launch errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "AMD kernel launch failed: %s\n", hipGetErrorString(err));
    }

#ifdef DEBUG_SHA1
    // Synchronize and check results in debug mode
    hipStreamSynchronize(config.stream);
    uint32_t result_count_host;
    hipMemcpy(&result_count_host, pool.count, sizeof(uint32_t), hipMemcpyDeviceToHost);
    printf("[DEBUG] AMD kernel completed. Results found: %u\n", result_count_host);
#endif
}
