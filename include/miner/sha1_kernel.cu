// sha1_mining_kernel.cu - Optimized SHA-1 near-collision mining kernel
#include "sha1_miner.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "utilities.hpp"

namespace cg = cooperative_groups;

// SHA-1 constants in constant memory
__device__ __constant__ uint32_t K[4] = {
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6
};

__device__ __constant__ uint32_t H0[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/**
 * Count leading zero bits between hash and target (XOR distance)
 */
__device__ __forceinline__ uint32_t count_leading_zeros_160bit(
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
            break; // Stop counting after first non-matching word
        }
    }
    return total_bits;
}

/**
 * Main SHA-1 mining kernel for NVIDIA GPUs
 * Processes multiple nonces per thread to find near-collisions
 */
__global__ void sha1_mining_kernel_nvidia(
    const uint8_t * __restrict__ base_message,
    const uint32_t * __restrict__ target_hash,
    uint32_t difficulty,
    MiningResult * __restrict__ results,
    uint32_t * __restrict__ result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread,
    uint64_t * __restrict__ actual_nonces_processed // ADD THIS for accurate tracking
) {
    // Thread indices
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // Track how many nonces this thread actually processes
    uint32_t nonces_processed = 0;

    // Load the base message as bytes
    uint8_t base_msg[32];
#pragma unroll
    for (int i = 0; i < 32; i++) {
        base_msg[i] = base_message[i];
    }

    // Load target (already in correct format)
    uint32_t target[5];
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = target_hash[i];
    }

    // Process nonces for this thread
    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0) continue;

        // Count this nonce as processed
        nonces_processed++;

        // Create a copy of the message
        uint8_t msg_bytes[32];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            ((uint32_t*)msg_bytes)[j] = ((uint32_t*)base_msg)[j];
        }

        // Apply nonce to last 8 bytes by XORing (big-endian)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            msg_bytes[24 + j] ^= (nonce >> (56 - j * 8)) & 0xFF;
        }

        // Convert message bytes to big-endian words for SHA-1
        uint32_t W[16];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = (static_cast<uint32_t>(msg_bytes[j * 4]) << 24) |
                   (static_cast<uint32_t>(msg_bytes[j * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg_bytes[j * 4 + 2]) << 8) |
                   static_cast<uint32_t>(msg_bytes[j * 4 + 3]);
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
                uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
                W[t & 15] = __funnelshift_l(temp, temp, 1);
            }
            uint32_t f = (b & c) | (~b & d);
            uint32_t temp = __funnelshift_l(a, a, 5) + f + e + K[0] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp;
        }

        // Rounds 20-39
#pragma unroll
        for (int t = 20; t < 40; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = __funnelshift_l(temp, temp, 1);
            uint32_t f = b ^ c ^ d;
            uint32_t temp2 = __funnelshift_l(a, a, 5) + f + e + K[1] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp2;
        }

        // Rounds 40-59
#pragma unroll
        for (int t = 40; t < 60; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = __funnelshift_l(temp, temp, 1);
            uint32_t f = (b & c) | (b & d) | (c & d);
            uint32_t temp2 = __funnelshift_l(a, a, 5) + f + e + K[2] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp2;
        }

        // Rounds 60-79
#pragma unroll
        for (int t = 60; t < 80; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = __funnelshift_l(temp, temp, 1);
            uint32_t f = b ^ c ^ d;
            uint32_t temp2 = __funnelshift_l(a, a, 5) + f + e + K[3] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
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
        uint32_t matching_bits = count_leading_zeros_160bit(hash, target);

#ifdef DEBUG_SHA1
        if (blockIdx.x == 0 && threadIdx.x == 0 && i == 0) {
            printf("[DEBUG] Thread 0: nonce=%llu, matching_bits=%u, difficulty=%u\n",
                   nonce, matching_bits, difficulty);
        }
#endif

        // If this meets difficulty, save it
        if (matching_bits >= difficulty) {
            uint32_t idx = atomicAdd(result_count, 1);
            if (idx < result_capacity) {
                results[idx].nonce = nonce;
                results[idx].matching_bits = matching_bits;
                results[idx].difficulty_score = matching_bits;
#pragma unroll
                for (int j = 0; j < 5; j++) {
                    results[idx].hash[j] = hash[j];
                }

#ifdef DEBUG_SHA1
                if (idx == 0) {
                    printf("[DEBUG] Found collision! nonce=%llu, bits=%u\n", nonce, matching_bits);
                }
#endif
            } else {
                atomicSub(result_count, 1);
            }
        }
    }

    // At end of kernel, atomically add the actual count of nonces processed
    atomicAdd((unsigned long long *) actual_nonces_processed, (unsigned long long) nonces_processed);
}

/**
 * Launch the SHA-1 mining kernel
 */
void launch_mining_kernel_nvidia(
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

    // Calculate shared memory size (if needed for future optimizations)
    uint32_t num_warps = (config.threads_per_block + 31) / 32;
    size_t uint32_part = sizeof(uint32_t) * (8 + 5 + num_warps);
    size_t aligned_offset = (uint32_part + 7) & ~7;
    size_t shared_mem_size = aligned_offset + sizeof(uint64_t) * num_warps;

    uint32_t nonces_per_thread = NONCES_PER_THREAD;

    // Reset result count before launching kernel
    cudaError_t err = cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to reset result count: %s\n", cudaGetErrorString(err));
        return;
    }

    // Clear previous errors
    cudaGetLastError();

    // Launch kernel
    dim3 gridDim(config.blocks, 1, 1);
    dim3 blockDim(config.threads_per_block, 1, 1);

#ifdef DEBUG_SHA1
    printf("[DEBUG] Launching kernel: blocks=%d, threads=%d, shared_mem=%zu bytes\n",
           config.blocks, config.threads_per_block, shared_mem_size);
    printf("[DEBUG] Mining parameters: difficulty=%u, nonce_offset=%llu, nonces_per_thread=%u\n",
           difficulty, nonce_offset, nonces_per_thread);
#endif

    sha1_mining_kernel_nvidia<<<gridDim, blockDim, shared_mem_size, config.stream>>>(
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
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

#ifdef DEBUG_SHA1
    // Synchronize and check results in debug mode
    cudaStreamSynchronize(config.stream);
    uint32_t result_count_host;
    cudaMemcpy(&result_count_host, pool.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("[DEBUG] Kernel completed. Results found: %u\n", result_count_host);
#endif
}
