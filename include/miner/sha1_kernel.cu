#include "sha1_miner.cuh"
#include <cuda_runtime.h>

// SHA-1 round constants
__device__ __constant__ uint32_t K[4] = {
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6
};

// Initial SHA-1 state
__device__ __constant__ uint32_t H[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

// Device constant memory for current job
__device__ __constant__ MiningJob d_job;

// SHA-1 round functions
__device__ __forceinline__ uint32_t f(int t, uint32_t b, uint32_t c, uint32_t d) {
    if (t < 20) return (b & c) | (~b & d);
    else if (t < 40 || t >= 60) return b ^ c ^ d;
    else return (b & c) | (b & d) | (c & d);
}

// OPTIMIZED SHA-1 implementation specifically for mining
__device__ __forceinline__ bool sha1_mining_fast(
    const uint8_t *message,
    uint64_t nonce,
    uint32_t *out_hash,
    uint32_t required_bits
) {
    // Message schedule - only 16 words needed
    uint32_t W[16];

    // Load and prepare message with nonce
#pragma unroll
    for (int i = 0; i < 8; i++) {
        W[i] = swap_endian(reinterpret_cast<const uint32_t *>(message)[i]);
    }

    // Apply nonce to words 6 and 7
    W[6] ^= swap_endian(static_cast<uint32_t>(nonce & 0xFFFFFFFF));
    W[7] ^= swap_endian(static_cast<uint32_t>(nonce >> 32));

    // Padding
    W[8] = 0x80000000;
#pragma unroll
    for (int i = 9; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 256; // Message length in bits

    // Initialize working variables
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4];

    // Main SHA-1 computation - optimized with partial unrolling
    // Rounds 0-15
#pragma unroll 16
    for (int t = 0; t < 16; t++) {
        uint32_t temp = rotl32(a, 5) + f(t, b, c, d) + e + K[0] + W[t];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;
    }

    // Rounds 16-79 with on-the-fly W computation
#pragma unroll 8  // Partial unroll for balance
    for (int t = 16; t < 80; t++) {
        W[t & 15] = rotl32(W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[t & 15], 1);
        uint32_t temp = rotl32(a, 5) + f(t, b, c, d) + e + K[t / 20] + W[t & 15];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;

        // Early exit check at rounds 40 and 60
        if ((t == 40 || t == 60) && required_bits > 64) {
            // Check if we're on track
            uint32_t partial_bits = count_matching_bits(a + H[0], d_job.target_hash[0]);
            if (t == 40 && partial_bits < 16) return false; // Too far off
            if (t == 60 && partial_bits < 24) return false; // Too far off
        }
    }

    // Final hash
    out_hash[0] = a + H[0];
    out_hash[1] = b + H[1];
    out_hash[2] = c + H[2];
    out_hash[3] = d + H[3];
    out_hash[4] = e + H[4];

    return true;
}

// Calculate total matching bits between two hashes
__device__ __forceinline__ uint32_t calculate_total_matching_bits(const uint32_t *hash, const uint32_t *target) {
    uint32_t total_bits = 0;

#pragma unroll
    for (int i = 0; i < 5; i++) {
        if (hash[i] == target[i]) {
            total_bits += 32;
        } else {
            total_bits += count_matching_bits(hash[i], target[i]);
            // Don't break - count all matching bits for better candidate selection
        }
    }

    return total_bits;
}

// Calculate consecutive matching bits (for difficulty score)
__device__ __forceinline__ uint32_t calculate_consecutive_bits(const uint32_t *hash, const uint32_t *target) {
    uint32_t consecutive = 0;

#pragma unroll
    for (int i = 0; i < 5; i++) {
        if (hash[i] == target[i]) {
            consecutive += 32;
        } else {
            consecutive += __clz(hash[i] ^ target[i]);
            break;
        }
    }

    return consecutive;
}

// Result structure for reduction
struct CandidateResult {
    uint64_t nonce;
    uint32_t matching_bits;
    uint32_t consecutive_bits;
    uint32_t hash[5];

    __device__ __forceinline__ void reset() {
        matching_bits = 0;
        consecutive_bits = 0;
        nonce = 0;
    }

    __device__ __forceinline__ bool is_better_than(const CandidateResult &other) const {
        if (consecutive_bits != other.consecutive_bits)
            return consecutive_bits > other.consecutive_bits;
        return matching_bits > other.matching_bits;
    }
};

// Main mining kernel - optimized for real-world performance
__global__ void sha1_near_collision_kernel(
    ResultPool pool,
    uint64_t nonce_base,
    uint32_t max_nonces
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t num_warps = blockDim.x / WARP_SIZE;

    // Shared memory for block-level reduction
    __shared__ CandidateResult s_warp_results[8]; // Max 8 warps per block
    __shared__ bool s_has_candidate;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        s_has_candidate = false;
    }
    if (threadIdx.x < num_warps) {
        s_warp_results[threadIdx.x].reset();
    }
    __syncthreads();

    // Local best candidate
    CandidateResult local_best;
    local_best.reset();

    // Process nonces
#pragma unroll 4  // Balance between register usage and efficiency
    for (int n = 0; n < NONCES_PER_THREAD; n++) {
        uint64_t nonce_idx = static_cast<uint64_t>(tid) * NONCES_PER_THREAD + n;
        if (nonce_idx >= max_nonces) break;

        uint64_t nonce = nonce_base + nonce_idx;
        uint32_t hash[5];

        // Compute SHA-1 with early exit
        if (!sha1_mining_fast(d_job.base_message, nonce, hash, d_job.difficulty)) {
            continue; // Early exit triggered
        }

        // Calculate matching metrics
        uint32_t total_bits = calculate_total_matching_bits(hash, d_job.target_hash);

        // Check if this meets difficulty threshold
        if (total_bits >= d_job.difficulty) {
            uint32_t consecutive = calculate_consecutive_bits(hash, d_job.target_hash);

            // Update local best if better
            CandidateResult candidate;
            candidate.nonce = nonce;
            candidate.matching_bits = total_bits;
            candidate.consecutive_bits = consecutive;
#pragma unroll
            for (int i = 0; i < 5; i++) {
                candidate.hash[i] = hash[i];
            }

            if (candidate.is_better_than(local_best)) {
                local_best = candidate;
            }
        }
    }

    // Warp-level reduction
    if (local_best.matching_bits >= d_job.difficulty) {
        // Reduce within warp using shuffle operations
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            CandidateResult other;
            other.nonce = __shfl_xor_sync(0xFFFFFFFF, local_best.nonce, offset);
            other.matching_bits = __shfl_xor_sync(0xFFFFFFFF, local_best.matching_bits, offset);
            other.consecutive_bits = __shfl_xor_sync(0xFFFFFFFF, local_best.consecutive_bits, offset);

#pragma unroll
            for (int i = 0; i < 5; i++) {
                other.hash[i] = __shfl_xor_sync(0xFFFFFFFF, local_best.hash[i], offset);
            }

            if (other.is_better_than(local_best)) {
                local_best = other;
            }
        }

        // Lane 0 writes to shared memory
        if (lane_id == 0) {
            s_warp_results[warp_id] = local_best;
            atomicOr(reinterpret_cast<int *>(&s_has_candidate), 1);
        }
    }
    __syncthreads();

    // Block-level reduction (first warp only)
    if (warp_id == 0 && s_has_candidate) {
        CandidateResult block_best;
        block_best.reset();

        // Load warp results
        if (lane_id < num_warps) {
            block_best = s_warp_results[lane_id];
        }

        // Reduce within first warp
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            if (lane_id < offset * 2) {
                CandidateResult other;
                other.matching_bits = __shfl_xor_sync(0xFF, block_best.matching_bits, offset);
                other.consecutive_bits = __shfl_xor_sync(0xFF, block_best.consecutive_bits, offset);
                other.nonce = __shfl_xor_sync(0xFF, block_best.nonce, offset);

#pragma unroll
                for (int i = 0; i < 5; i++) {
                    other.hash[i] = __shfl_xor_sync(0xFF, block_best.hash[i], offset);
                }

                if (other.is_better_than(block_best)) {
                    block_best = other;
                }
            }
        }

        // Thread 0 writes final result
        if (lane_id == 0 && block_best.matching_bits >= d_job.difficulty) {
            uint32_t idx = atomicAdd(pool.count, 1);
            if (idx < pool.capacity) {
                MiningResult &result = pool.results[idx];
                result.nonce = block_best.nonce;
                result.matching_bits = block_best.matching_bits;
                result.difficulty_score = block_best.consecutive_bits;
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    result.hash[i] = block_best.hash[i];
                }
            }
        }
    }
}

// Host-side kernel launcher
extern "C" void launch_mining_kernel(
    const MiningJob &job,
    ResultPool &pool,
    const KernelConfig &config
) {
    // Upload job to constant memory
    cudaMemcpyToSymbolAsync(d_job, &job, sizeof(MiningJob), 0,
                            cudaMemcpyHostToDevice, config.stream);

    // Calculate work distribution
    uint64_t total_nonces = static_cast<uint64_t>(config.blocks) *
                            static_cast<uint64_t>(config.threads_per_block) *
                            static_cast<uint64_t>(NONCES_PER_THREAD);

    // Launch kernel
    sha1_near_collision_kernel<<<config.blocks, config.threads_per_block,
            0, config.stream>>>(
        pool, job.nonce_offset, total_nonces
    );

#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
#endif
}