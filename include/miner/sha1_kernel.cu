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
    volatile JobUpdateRequest * __restrict__ job_update,
    uint64_t * __restrict__ current_job_version
) {
    // Thread indices
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & 31;
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // Check if live updates are enabled
    const bool live_updates_enabled = (job_update != nullptr && current_job_version != nullptr);

    // Shared memory for job data
    __shared__ uint8_t shared_base_msg[32];
    __shared__ uint32_t shared_target[5];
    __shared__ uint32_t shared_difficulty;
    __shared__ uint64_t shared_job_version;

    // Initialize shared memory - load job version first
    if (threadIdx.x == 0) {
        shared_job_version = live_updates_enabled ? *current_job_version : 0;
    }
    __syncthreads();

    // Load initial job data
    if (threadIdx.x < 8) {
        ((uint4*)shared_base_msg)[threadIdx.x] = ((const uint4*)base_message)[threadIdx.x];
    }
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = target_hash[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        shared_difficulty = difficulty;
    }
    __syncthreads();

    // Track processed nonces
    uint32_t nonces_processed = 0;

    // Main mining loop
    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        // Check for job updates periodically
        if (live_updates_enabled && (i & 0x3FF) == 0) {  // Every 1024 iterations
            // All threads participate in checking
            __syncthreads();

            // One thread checks for update
            __shared__ bool needs_update;
            if (threadIdx.x == 0) {
                needs_update = false;
                // Check if there's a new job version available
                if (job_update->job_updated) {
                    uint64_t new_version = job_update->job_version;
                    if (new_version > shared_job_version) {
                        needs_update = true;
                        shared_job_version = new_version;
                    }
                }
            }
            __syncthreads();

            // If update needed, all threads load new data
            if (needs_update) {
                // Load new job data
                if (threadIdx.x < 8) {
                    ((uint4*)shared_base_msg)[threadIdx.x] = ((uint4*)job_update->base_message)[threadIdx.x];
                }
                if (threadIdx.x < 5) {
                    shared_target[threadIdx.x] = job_update->target_hash[threadIdx.x];
                }
                if (threadIdx.x == 0) {
                    shared_difficulty = job_update->difficulty;
                    // Update the current job version for this kernel
                    *current_job_version = shared_job_version;
                    // DO NOT reset result count here - let host handle it
                    // DO NOT clear job_updated flag here - let host handle it
                }
                __syncthreads();
            }
        }

        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0) continue;

        nonces_processed++;

        // Create message copy from shared memory
        uint8_t msg_bytes[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            msg_bytes[j] = shared_base_msg[j];
        }

        // Apply nonce
        uint32_t* msg_words = (uint32_t*)msg_bytes;
        msg_words[6] ^= __byte_perm(nonce >> 32, 0, 0x0123);
        msg_words[7] ^= __byte_perm(nonce & 0xFFFFFFFF, 0, 0x0123);

        // Convert to big-endian words for SHA-1
        uint32_t W[16];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = __byte_perm(msg_words[j], 0, 0x0123);
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
        #pragma unroll
        for (int t = 0; t < 20; t++) {
            if (t >= 16) {
                W[t & 15] = __funnelshift_l(W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                           W[(t-14) & 15] ^ W[(t-16) & 15],
                                           W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                           W[(t-14) & 15] ^ W[(t-16) & 15], 1);
            }
            uint32_t temp = __funnelshift_l(a, a, 5) + ((b & c) | (~b & d)) + e + K[0] + W[t & 15];
            e = d; d = c; c = __funnelshift_l(b, b, 30); b = a; a = temp;
        }

        // Rounds 20-39
        #pragma unroll
        for (int t = 20; t < 40; t++) {
            W[t & 15] = __funnelshift_l(W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                       W[(t-14) & 15] ^ W[(t-16) & 15],
                                       W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                       W[(t-14) & 15] ^ W[(t-16) & 15], 1);
            uint32_t temp = __funnelshift_l(a, a, 5) + (b ^ c ^ d) + e + K[1] + W[t & 15];
            e = d; d = c; c = __funnelshift_l(b, b, 30); b = a; a = temp;
        }

        // Rounds 40-59
        #pragma unroll
        for (int t = 40; t < 60; t++) {
            W[t & 15] = __funnelshift_l(W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                       W[(t-14) & 15] ^ W[(t-16) & 15],
                                       W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                       W[(t-14) & 15] ^ W[(t-16) & 15], 1);
            uint32_t temp = __funnelshift_l(a, a, 5) + ((b & c) | (d & (b ^ c))) + e + K[2] + W[t & 15];
            e = d; d = c; c = __funnelshift_l(b, b, 30); b = a; a = temp;
        }

        // Rounds 60-79
        #pragma unroll
        for (int t = 60; t < 80; t++) {
            W[t & 15] = __funnelshift_l(W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                       W[(t-14) & 15] ^ W[(t-16) & 15],
                                       W[(t-3) & 15] ^ W[(t-8) & 15] ^
                                       W[(t-14) & 15] ^ W[(t-16) & 15], 1);
            uint32_t temp = __funnelshift_l(a, a, 5) + (b ^ c ^ d) + e + K[3] + W[t & 15];
            e = d; d = c; c = __funnelshift_l(b, b, 30); b = a; a = temp;
        }

        // Add initial hash values
        uint32_t hash[5];
        hash[0] = a + H0[0];
        hash[1] = b + H0[1];
        hash[2] = c + H0[2];
        hash[3] = d + H0[3];
        hash[4] = e + H0[4];

        // Count matching bits
        uint32_t matching_bits = count_leading_zeros_160bit(hash, shared_target);

        // Check if we found a match
        if (matching_bits >= shared_difficulty) {
            // Use warp vote functions for efficient result writing
            unsigned mask = __ballot_sync(0xffffffff, matching_bits >= shared_difficulty);

            if (mask != 0) {
                // Count matches before this lane
                unsigned lane_mask = (1U << lane_id) - 1;
                unsigned prefix_sum = __popc(mask & lane_mask);

                // First active lane reserves space for all matches
                unsigned base_idx;
                if (lane_id == __ffs(mask) - 1) {
                    base_idx = atomicAdd(result_count, __popc(mask));
                }

                // Broadcast base index to all lanes
                base_idx = __shfl_sync(0xffffffff, base_idx, __ffs(mask) - 1);

                // Each matching lane writes its result
                if ((mask >> lane_id) & 1) {
                    unsigned idx = base_idx + prefix_sum;
                    if (idx < result_capacity) {
                        results[idx].nonce = nonce;
                        results[idx].matching_bits = matching_bits;
                        results[idx].difficulty_score = matching_bits;
                        results[idx].job_version = shared_job_version;
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
    const KernelConfig &config
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

    // Debug print the pointer values
    printf("[DEBUG] Pool pointers - count=%p, results=%p, nonces=%p, job_version=%p\n",
           pool.count, pool.results, pool.nonces_processed, pool.job_version);

    printf("[DEBUG] Launching SHA-1 mining kernel with live job update support...\n");
    printf("[DEBUG] Launching mining kernel with difficulty=%u, nonce_offset=%llu\n",
           difficulty, nonce_offset);

    // Validate job update pointer
    if (!device_job.job_update) {
        fprintf(stderr, "Warning: job_update pointer is null, live updates disabled\n");
    }

    // Validate pool.job_version pointer
    if (!pool.job_version) {
        fprintf(stderr, "Error: pool.job_version is null\n");
        return;
    }

    // No shared memory needed for simplified kernel
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

    // Launch without job update functionality for now
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
        device_job.job_update,
        pool.job_version
    );

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}