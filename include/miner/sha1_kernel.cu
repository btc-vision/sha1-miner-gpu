#include "sha1_miner.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

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
    uint64_t * __restrict__ actual_nonces_processed,
    uint64_t job_version,
    uint32_t stream_id,
    uint32_t total_blocks_per_stream,
    uint64_t *assigned_nonces,
    uint32_t *conflict_counter
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & 31;

    // Direct nonce calculation
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // Check if this range was already assigned
    if (assigned_nonces && conflict_counter) {
        uint64_t range_id = thread_nonce_base / nonces_per_thread;
        uint64_t word_idx = range_id / 64;
        uint64_t bit_idx = range_id % 64;

        // IMPORTANT: Bounds check to prevent memory corruption
        uint64_t max_words = 134217728; // Adjust based on your allocation (1GB / 8)
        if (word_idx >= max_words) {
            // Out of bounds - skip this thread
            return;
        }

        // Atomic check and set
        uint64_t old_val = atomicOr(&assigned_nonces[word_idx], 1ULL << bit_idx);
        if (old_val & (1ULL << bit_idx)) {
            // This range was already assigned
            atomicAdd(conflict_counter, 1);
            // Don't print from every thread - too much output
            if (tid == 0) {
                printf("CONFLICT detected at range %llu\n", range_id);
            }
            return;
        }
    }

    // Load base message using vectorized access
    uint8_t base_msg[32];
    uint4 *base_msg_vec = (uint4 *) base_msg;
    const uint4 *base_message_vec = (const uint4 *) base_message;
    base_msg_vec[0] = base_message_vec[0];
    base_msg_vec[1] = base_message_vec[1];

    // Load target
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

        // Create message copy with vectorized ops
        uint8_t msg_bytes[32];
        uint4 *msg_bytes_vec = (uint4 *) msg_bytes;
        msg_bytes_vec[0] = base_msg_vec[0];
        msg_bytes_vec[1] = base_msg_vec[1];

        // Apply nonce efficiently
        uint32_t *msg_words = (uint32_t *) msg_bytes;
        msg_words[6] ^= __byte_perm(nonce >> 32, 0, 0x0123);
        msg_words[7] ^= __byte_perm(nonce & 0xFFFFFFFF, 0, 0x0123);

        // Convert to big-endian words for SHA-1
        uint32_t W[16];
        uint32_t *msg_words_local = (uint32_t *) msg_bytes;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = __byte_perm(msg_words_local[j], 0, 0x0123);
        }

        // Apply SHA-1 padding
        W[8] = 0x80000000;
#pragma unroll
        for (int j = 9; j < 15; j++) {
            W[j] = 0;
        }
        W[15] = 0x00000100; // Message length (256 bits)

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
                W[t & 15] = __funnelshift_l(W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                            W[(t - 14) & 15] ^ W[(t - 16) & 15],
                                            W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                            W[(t - 14) & 15] ^ W[(t - 16) & 15], 1);
            }
            uint32_t temp = __funnelshift_l(a, a, 5) + ((b & c) | (~b & d)) + e + K[0] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp;
        }

        // Rounds 20-39
#pragma unroll
        for (int t = 20; t < 40; t++) {
            W[t & 15] = __funnelshift_l(W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                        W[(t - 14) & 15] ^ W[(t - 16) & 15],
                                        W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                        W[(t - 14) & 15] ^ W[(t - 16) & 15], 1);
            uint32_t temp = __funnelshift_l(a, a, 5) + (b ^ c ^ d) + e + K[1] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp;
        }

        // Rounds 40-59
#pragma unroll
        for (int t = 40; t < 60; t++) {
            W[t & 15] = __funnelshift_l(W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                        W[(t - 14) & 15] ^ W[(t - 16) & 15],
                                        W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                        W[(t - 14) & 15] ^ W[(t - 16) & 15], 1);
            uint32_t temp = __funnelshift_l(a, a, 5) + ((b & c) | (d & (b ^ c))) + e + K[2] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp;
        }

        // Rounds 60-79
#pragma unroll
        for (int t = 60; t < 80; t++) {
            W[t & 15] = __funnelshift_l(W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                        W[(t - 14) & 15] ^ W[(t - 16) & 15],
                                        W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                                        W[(t - 14) & 15] ^ W[(t - 16) & 15], 1);
            uint32_t temp = __funnelshift_l(a, a, 5) + (b ^ c ^ d) + e + K[3] + W[t & 15];
            e = d;
            d = c;
            c = __funnelshift_l(b, b, 30);
            b = a;
            a = temp;
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

        // Check if we found a match
        if (matching_bits >= difficulty) {
            // Use warp vote functions for efficient result writing
            unsigned mask = __ballot_sync(0xffffffff, matching_bits >= difficulty);

            if (mask != 0) {
                // Count matches before this lane
                unsigned lane_mask = (1U << lane_id) - 1;
                unsigned prefix_sum = __popc(mask & lane_mask);

                // First active lane reserves space
                unsigned base_idx;
                if (lane_id == __ffs(mask) - 1) {
                    base_idx = atomicAdd(result_count, __popc(mask));
                }

                // Broadcast base index
                base_idx = __shfl_sync(0xffffffff, base_idx, __ffs(mask) - 1);

                // Write result
                if ((mask >> lane_id) & 1) {
                    unsigned idx = base_idx + prefix_sum;
                    if (idx < result_capacity) {
                        results[idx].nonce = nonce;
                        results[idx].matching_bits = matching_bits;
                        results[idx].difficulty_score = matching_bits;
                        results[idx].job_version = job_version;
#pragma unroll
                        for (int j = 0; j < 5; j++) {
                            results[idx].hash[j] = hash[j];
                        }
                    }
                }
            }
        }
    }

    // Update total nonces processed
    atomicAdd(actual_nonces_processed, nonces_processed);
}

/**
 * Launch the SHA-1 mining kernel
 */
void launch_mining_kernel_nvidia(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version,
    uint32_t stream_id,
    uint64_t *assigned_nonces,
    uint32_t *conflict_counter
) {
    // Validate configuration
    if (config.blocks <= 0 || config.threads_per_block <= 0) {
        fprintf(stderr, "Invalid kernel configuration: blocks=%d, threads=%d\n",
                config.blocks, config.threads_per_block);
        return;
    }

    // Validate pool pointers
    if (!pool.results || !pool.count || !pool.nonces_processed) {
        fprintf(stderr, "ERROR: Invalid pool pointers - results=%p, count=%p, nonces=%p\n",
                pool.results, pool.count, pool.nonces_processed);
        return;
    }

    // No shared memory needed
    size_t shared_mem_size = 0;
    uint32_t nonces_per_thread = NONCES_PER_THREAD;

    // Reset result count before launching kernel
    cudaError_t err = cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to reset result count: %s (pointer=%p)\n",
                cudaGetErrorString(err), pool.count);
        return;
    }

    // Clear previous errors
    cudaGetLastError();

    // Launch kernel
    dim3 gridDim(config.blocks, 1, 1);
    dim3 blockDim(config.threads_per_block, 1, 1);

    sha1_mining_kernel_nvidia<<<gridDim, blockDim, shared_mem_size, config.stream>>>(
        device_job.base_message,
        device_job.target_hash,
        difficulty,
        pool.results,
        pool.count,
        pool.capacity,
        nonce_offset,
        nonces_per_thread,
        pool.nonces_processed,
        job_version,
        stream_id,
        config.blocks,
        assigned_nonces,
        conflict_counter
    );

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
