#include "sha1_miner_core.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Optimized SHA-1 round functions with early exit checks
__device__ __forceinline__ bool sha1_compress_with_early_exit(
    uint32_t W[16],
    uint32_t state[5],
    const uint32_t target[5],
    uint32_t min_matching_bits,
    uint32_t check_round = 60  // Start checking after round 60
) {
    uint32_t a = SHA1_H0;
    uint32_t b = SHA1_H1;
    uint32_t c = SHA1_H2;
    uint32_t d = SHA1_H3;
    uint32_t e = SHA1_H4;

    // Unroll first 20 rounds (no early exit here, too early)
    #pragma unroll 20
    for (int t = 0; t < 20; t++) {
        uint32_t f = (b & c) | (~b & d);
        uint32_t temp = rotl32(a, 5) + f + e + SHA1_K0 + W[t];
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }

    // Rounds 20-40
    #pragma unroll 20
    for (int t = 20; t < 40; t++) {
        W[t & 15] = rotl32(W[(t-3)&15] ^ W[(t-8)&15] ^ W[(t-14)&15] ^ W[(t-16)&15], 1);
        uint32_t f = b ^ c ^ d;
        uint32_t temp = rotl32(a, 5) + f + e + SHA1_K1 + W[t & 15];
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }

    // Rounds 40-60
    #pragma unroll 20
    for (int t = 40; t < 60; t++) {
        W[t & 15] = rotl32(W[(t-3)&15] ^ W[(t-8)&15] ^ W[(t-14)&15] ^ W[(t-16)&15], 1);
        uint32_t f = (b & c) | (b & d) | (c & d);
        uint32_t temp = rotl32(a, 5) + f + e + SHA1_K2 + W[t & 15];
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }

    // Early exit check - can we possibly meet the target?
    // This is based on the observation that the final 20 rounds have limited mixing
    if (check_round == 60) {
        // Rough approximation of final state
        uint32_t approx_a = a + SHA1_H0;
        uint32_t approx_b = b + SHA1_H1;

        // Check if we're on track for the required difficulty
        uint32_t early_diff = popcount32(approx_a ^ target[0]) + popcount32(approx_b ^ target[1]);
        if (early_diff > (160 - min_matching_bits) * 2 / 5) {
            return false;  // Early exit - unlikely to meet difficulty
        }
    }

    // Rounds 60-80
    #pragma unroll 20
    for (int t = 60; t < 80; t++) {
        W[t & 15] = rotl32(W[(t-3)&15] ^ W[(t-8)&15] ^ W[(t-14)&15] ^ W[(t-16)&15], 1);
        uint32_t f = b ^ c ^ d;
        uint32_t temp = rotl32(a, 5) + f + e + SHA1_K3 + W[t & 15];
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }

    // Final addition
    state[0] = a + SHA1_H0;
    state[1] = b + SHA1_H1;
    state[2] = c + SHA1_H2;
    state[3] = d + SHA1_H3;
    state[4] = e + SHA1_H4;

    return true;
}

// Main collision mining kernel
extern "C" __global__ __launch_bounds__(256, 4)
void sha1_collision_mine_kernel(
    CollisionCandidate* __restrict__ candidates,
    uint32_t* __restrict__ candidate_count,
    uint32_t* __restrict__ best_distance,
    const uint64_t total_threads
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;

    // Shared memory for warp-level candidate collection
    __shared__ CollisionCandidate warp_candidates[8][MAX_CANDIDATES_PER_BLOCK];
    __shared__ uint32_t warp_candidate_count[8];
    __shared__ uint32_t warp_best_distance[8];

    if (threadIdx.x < 8) {
        warp_candidate_count[threadIdx.x] = 0;
        warp_best_distance[threadIdx.x] = 160;  // Worst case
    }
    __syncthreads();

    // Calculate nonce range for this thread
    uint64_t nonces_per_thread = g_job.nonce_range / total_threads;
    uint64_t my_nonce_start = g_job.nonce_start + tid * nonces_per_thread;
    uint64_t my_nonce_end = my_nonce_start + nonces_per_thread;

    // Process multiple nonces per thread
    const uint32_t NONCES_PER_ITERATION = 8;

    for (uint64_t base_nonce = my_nonce_start; base_nonce < my_nonce_end; base_nonce += NONCES_PER_ITERATION) {
        #pragma unroll 4
        for (uint32_t n = 0; n < NONCES_PER_ITERATION && (base_nonce + n) < my_nonce_end; n++) {
            uint64_t nonce = base_nonce + n;

            // Prepare message with nonce
            uint32_t W[16];
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                W[i] = bswap32(g_job.base_msg[i]);
            }
            W[6] = bswap32(g_job.base_msg[6] ^ (uint32_t)(nonce & 0xFFFFFFFF));
            W[7] = bswap32(g_job.base_msg[7] ^ (uint32_t)(nonce >> 32));
            W[8] = 0x80000000;  // Padding
            #pragma unroll
            for (int i = 9; i < 15; i++) {
                W[i] = 0;
            }
            W[15] = 256;  // Message length in bits

            // Compute SHA-1 with early exit
            uint32_t hash[5];
            bool completed = sha1_compress_with_early_exit(
                W, hash, g_job.target_hash, g_job.difficulty_bits
            );

            if (!completed) continue;  // Early exit triggered

            // Check if this meets our difficulty requirement
            uint32_t matching_bits = count_matching_bits(hash, g_job.target_hash);

            if (matching_bits >= g_job.difficulty_bits) {
                // Found a candidate! Use atomic to get a slot
                uint32_t my_slot = atomicAdd(&warp_candidate_count[warp_id], 1);

                if (my_slot < MAX_CANDIDATES_PER_BLOCK) {
                    CollisionCandidate& cand = warp_candidates[warp_id][my_slot];
                    cand.nonce = nonce;
                    #pragma unroll
                    for (int i = 0; i < 5; i++) {
                        cand.hash[i] = hash[i];
                    }
                    cand.distance_bits = 160 - matching_bits;
                    cand.thread_id = tid;

                    // Update best distance
                    atomicMin(&warp_best_distance[warp_id], cand.distance_bits);
                }

                // If we found an exact collision, we can stop
                if (matching_bits == 160) {
                    return;
                }
            }
        }
    }

    __syncthreads();

    // Warp 0 collects results from all warps
    if (warp_id == 0 && lane_id < 8) {
        uint32_t total_candidates = 0;
        for (int w = 0; w < 8; w++) {
            total_candidates += warp_candidate_count[w];
        }

        if (total_candidates > 0 && lane_id == 0) {
            uint32_t base_idx = atomicAdd(candidate_count, total_candidates);

            // Copy candidates to global memory
            uint32_t global_idx = base_idx;
            for (int w = 0; w < 8; w++) {
                for (uint32_t c = 0; c < warp_candidate_count[w] && c < MAX_CANDIDATES_PER_BLOCK; c++) {
                    if (global_idx < CANDIDATES_RING_SIZE) {
                        candidates[global_idx] = warp_candidates[w][c];
                        atomicMin(best_distance, warp_candidates[w][c].distance_bits);
                    }
                    global_idx++;
                }
            }
        }
    }
}

// Kernel for filtering candidates by difficulty before PCIe transfer
extern "C" __global__ void filter_candidates_kernel(
    const CollisionCandidate* __restrict__ all_candidates,
    CollisionCandidate* __restrict__ filtered_candidates,
    const uint32_t total_candidates,
    uint32_t* __restrict__ filtered_count,
    const uint32_t max_distance_bits
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_candidates) {
        const CollisionCandidate& cand = all_candidates[tid];

        // Only keep candidates that meet our threshold
        if (cand.distance_bits <= max_distance_bits) {
            uint32_t idx = atomicAdd(filtered_count, 1);
            if (idx < CANDIDATES_RING_SIZE / 4) {  // Keep top 25%
                filtered_candidates[idx] = cand;
            }
        }
    }
}