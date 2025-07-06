#include "sha1_miner.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>  // Optional: for advanced reductions

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

// Structure to hold candidate result for reduction
struct CandidateResult {
    uint64_t nonce;
    uint32_t matching_bits;
    uint32_t difficulty_score;
    uint32_t hash[5];
    uint32_t thread_id; // To track which thread has the result

    __device__ __forceinline__ bool operator>(const CandidateResult &other) const {
        if (matching_bits != other.matching_bits) {
            return matching_bits > other.matching_bits;
        }
        return difficulty_score > other.difficulty_score;
    }
};

// SHA-1 round functions
__device__ __forceinline__ uint32_t f(int t, uint32_t b, uint32_t c, uint32_t d) {
    if (t < 20) return (b & c) | (~b & d);
    else if (t < 40 || t >= 60) return b ^ c ^ d;
    else return (b & c) | (b & d) | (c & d);
}

// Optimized SHA-1 implementation with early exit
__device__ bool sha1_with_early_exit(
    const uint8_t *message,
    uint64_t nonce,
    uint32_t *out_hash,
    uint32_t *out_matching_bits,
    uint32_t required_bits
) {
    // Prepare message with nonce
    uint32_t W[80];

    // Load message into W[0-7]
#pragma unroll
    for (int i = 0; i < 6; i++) {
        W[i] = swap_endian(reinterpret_cast<const uint32_t *>(message)[i]);
    }

    // Apply nonce to last 8 bytes
    W[6] = swap_endian(reinterpret_cast<const uint32_t *>(message)[6] ^ static_cast<uint32_t>(nonce & 0xFFFFFFFF));
    W[7] = swap_endian(reinterpret_cast<const uint32_t *>(message)[7] ^ static_cast<uint32_t>(nonce >> 32));

    // Padding
    W[8] = 0x80000000;
#pragma unroll
    for (int i = 9; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 256; // Message length in bits

    // Message expansion - partially unroll for better performance
#pragma unroll 16
    for (int i = 16; i < 80; i++) {
        W[i] = rotl32(W[i - 3] ^ W[i - 8] ^ W[i - 14] ^ W[i - 16], 1);
    }

    // Initialize working variables
    uint32_t a = H[0];
    uint32_t b = H[1];
    uint32_t c = H[2];
    uint32_t d = H[3];
    uint32_t e = H[4];

    // Main SHA-1 loop with early exit checks
#pragma unroll 4
    for (int t = 0; t < 80; t++) {
        uint32_t temp = rotl32(a, 5) + f(t, b, c, d) + e + K[t / 20] + W[t];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;

        // Early exit check every 20 rounds
        if ((t == 19 || t == 39 || t == 59) && required_bits > 32) {
            // Quick check on first word only
            uint32_t partial_hash = a + H[0];
            uint32_t partial_bits = count_matching_bits(partial_hash, d_job.target_hash[0]);

            // If we can't even match enough bits in the first word, exit early
            if (partial_bits < min(32u, required_bits)) {
                return false;
            }
        }
    }

    // Final hash values
    out_hash[0] = a + H[0];
    out_hash[1] = b + H[1];
    out_hash[2] = c + H[2];
    out_hash[3] = d + H[3];
    out_hash[4] = e + H[4];

    // Calculate matching bits
    *out_matching_bits = 0;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        *out_matching_bits += count_matching_bits(out_hash[i], d_job.target_hash[i]);
        // Early termination if we've found a mismatch
        if (out_hash[i] != d_job.target_hash[i]) {
            *out_matching_bits += __clz(out_hash[i] ^ d_job.target_hash[i]);
            break;
        }
    }

    return *out_matching_bits >= required_bits;
}

// Advanced difficulty scoring for near-collisions
__device__ uint32_t calculate_difficulty_score(const uint32_t *hash, const uint32_t *target) {
    uint32_t score = 0;

    // Count consecutive matching bits from MSB
#pragma unroll
    for (int word = 0; word < 5; word++) {
        uint32_t diff = hash[word] ^ target[word];
        if (diff == 0) {
            score += 32;
        } else {
            score += __clz(diff);
            break;
        }
    }

    return score;
}

// Main mining kernel with full block-level reduction
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
    __shared__ CandidateResult s_warp_results[8]; // Max 8 warps per block (256 threads)
    __shared__ CandidateResult s_block_result; // Final block result
    __shared__ uint32_t s_has_candidate; // Flag indicating if block has a valid candidate

    // Initialize shared memory
    if (threadIdx.x == 0) {
        s_has_candidate = 0;
    }
    if (threadIdx.x < num_warps) {
        s_warp_results[threadIdx.x].matching_bits = 0;
        s_warp_results[threadIdx.x].difficulty_score = 0;
        s_warp_results[threadIdx.x].thread_id = 0xFFFFFFFF;
    }
    __syncthreads();

    // Local best result for this thread
    CandidateResult local_best;
    local_best.matching_bits = 0;
    local_best.difficulty_score = 0;
    local_best.thread_id = threadIdx.x;
    local_best.nonce = 0;

    // Process multiple nonces per thread
    for (int n = 0; n < NONCES_PER_THREAD; n++) {
        uint64_t nonce_idx = static_cast<uint64_t>(tid) * NONCES_PER_THREAD + n;
        if (nonce_idx >= max_nonces) break;

        uint64_t nonce = nonce_base + nonce_idx;

        uint32_t hash[5];
        uint32_t matching_bits;

        // Compute SHA-1 with early exit
        bool meets_difficulty = sha1_with_early_exit(
            d_job.base_message,
            nonce,
            hash,
            &matching_bits,
            d_job.difficulty
        );

        // Update local best if this is better
        if (matching_bits > local_best.matching_bits) {
            local_best.matching_bits = matching_bits;
            local_best.nonce = nonce;
            local_best.difficulty_score = calculate_difficulty_score(hash, d_job.target_hash);
#pragma unroll
            for (int i = 0; i < 5; i++) {
                local_best.hash[i] = hash[i];
            }
        } else if (matching_bits == local_best.matching_bits && meets_difficulty) {
            uint32_t score = calculate_difficulty_score(hash, d_job.target_hash);
            if (score > local_best.difficulty_score) {
                local_best.difficulty_score = score;
                local_best.nonce = nonce;
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    local_best.hash[i] = hash[i];
                }
            }
        }
    }

    // Step 1: Warp-level reduction using shuffle operations
    CandidateResult warp_best = local_best;

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        CandidateResult other;
        other.matching_bits = __shfl_xor_sync(0xFFFFFFFF, warp_best.matching_bits, offset);
        other.difficulty_score = __shfl_xor_sync(0xFFFFFFFF, warp_best.difficulty_score, offset);
        other.nonce = __shfl_xor_sync(0xFFFFFFFF, warp_best.nonce, offset);
        other.thread_id = __shfl_xor_sync(0xFFFFFFFF, warp_best.thread_id, offset);

        // Also shuffle the hash
#pragma unroll
        for (int i = 0; i < 5; i++) {
            other.hash[i] = __shfl_xor_sync(0xFFFFFFFF, warp_best.hash[i], offset);
        }

        if (other > warp_best) {
            warp_best = other;
        }
    }

    // Step 2: Lane 0 writes warp's best to shared memory
    if (lane_id == 0 && warp_best.matching_bits >= d_job.difficulty) {
        s_warp_results[warp_id] = warp_best;
        atomicOr(&s_has_candidate, 1); // Signal that we have at least one candidate
    }
    __syncthreads();

    // Step 3: Block-level reduction (only first warp participates)
    if (warp_id == 0 && s_has_candidate > 0) {
        CandidateResult block_best;
        block_best.matching_bits = 0;

        // Each thread in first warp loads one warp result
        if (lane_id < num_warps) {
            block_best = s_warp_results[lane_id];
        }

        // Reduce within first warp
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            if (lane_id < offset * 2) {
                CandidateResult other;
                other.matching_bits = __shfl_xor_sync(0xFF, block_best.matching_bits, offset);
                other.difficulty_score = __shfl_xor_sync(0xFF, block_best.difficulty_score, offset);
                other.nonce = __shfl_xor_sync(0xFF, block_best.nonce, offset);

#pragma unroll
                for (int i = 0; i < 5; i++) {
                    other.hash[i] = __shfl_xor_sync(0xFF, block_best.hash[i], offset);
                }

                if (other > block_best) {
                    block_best = other;
                }
            }
        }

        // Thread 0 has the block's best result
        if (lane_id == 0) {
            s_block_result = block_best;
        }
    }
    __syncthreads();

    // Step 4: Thread 0 writes the block's best result to global memory
    if (threadIdx.x == 0 && s_has_candidate > 0 && s_block_result.matching_bits >= d_job.difficulty) {
        uint32_t result_idx = atomicAdd(pool.count, 1);

        if (result_idx < pool.capacity) {
            MiningResult &result = pool.results[result_idx];
            result.nonce = s_block_result.nonce;
            result.matching_bits = s_block_result.matching_bits;
            result.difficulty_score = s_block_result.difficulty_score;

#pragma unroll
            for (int i = 0; i < 5; i++) {
                result.hash[i] = s_block_result.hash[i];
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
            config.shared_memory_size, config.stream>>>(
                pool, job.nonce_offset, total_nonces
            );

    // Check for launch errors in debug mode
#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
#endif
}
