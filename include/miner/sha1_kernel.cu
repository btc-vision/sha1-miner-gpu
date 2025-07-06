#include "sha1_miner.cuh"

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

// Optimized SHA-1 implementation with early exit
__device__ bool sha1_with_early_exit(
    const uint8_t* message,
    uint64_t nonce,
    uint32_t* out_hash,
    uint32_t* out_matching_bits,
    uint32_t required_bits
) {
    // Prepare message with nonce
    uint32_t W[80];

    // Load message into W[0-7]
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        W[i] = swap_endian(((uint32_t*)message)[i]);
    }

    // Apply nonce to last 8 bytes
    W[6] = swap_endian(((uint32_t*)message)[6] ^ (uint32_t)(nonce & 0xFFFFFFFF));
    W[7] = swap_endian(((uint32_t*)message)[7] ^ (uint32_t)(nonce >> 32));

    // Padding
    W[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 256; // Message length in bits

    // Message expansion
    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = rotl32(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);
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
        uint32_t temp = rotl32(a, 5) + f(t, b, c, d) + e + K[t/20] + W[t];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = temp;

        // Early exit check every 20 rounds
        if (t == 39 || t == 59) {
            uint32_t state[5] = {a + H[0], b + H[1], c + H[2], d + H[3], e + H[4]};
            uint32_t partial_bits = 0;

            #pragma unroll
            for (int i = 0; i < 5; i++) {
                partial_bits += count_matching_bits(state[i], d_job.target_hash[i]);
            }

            // Estimate if we can still reach required bits
            uint32_t rounds_left = 80 - t - 1;
            uint32_t max_possible_bits = partial_bits + (rounds_left / 4); // Heuristic

            if (max_possible_bits < required_bits) {
                return false; // Early exit
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
    }

    return *out_matching_bits >= required_bits;
}

// Advanced difficulty scoring for near-collisions
__device__ uint32_t calculate_difficulty_score(const uint32_t* hash, const uint32_t* target) {
    uint32_t score = 0;

    // Count consecutive matching bits from MSB
    for (int word = 0; word < 5; word++) {
        uint32_t diff = hash[word] ^ target[word];
        if (diff == 0) {
            score += 32;
        } else {
            score += __clz(diff); // Count leading zeros
            break;
        }
    }

    return score;
}

// Main mining kernel - optimized for near-collision detection
__global__ void sha1_near_collision_kernel(
    ResultPool pool,
    uint64_t nonce_base,
    uint32_t max_nonces
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);

    // Shared memory for warp-level result aggregation
    __shared__ uint32_t s_best_matches[8]; // One per warp in a 256-thread block
    __shared__ uint32_t s_best_nonces[8];
    __shared__ uint32_t s_best_scores[8];

    if (threadIdx.x < 8) {
        s_best_matches[threadIdx.x] = 0;
        s_best_scores[threadIdx.x] = 0;
    }
    __syncthreads();

    // Local variables for best result tracking
    uint32_t local_best_match = 0;
    uint64_t local_best_nonce = 0;
    uint32_t local_best_score = 0;
    uint32_t local_best_hash[5];

    // Process multiple nonces per thread
    #pragma unroll
    for (int n = 0; n < NONCES_PER_THREAD; n++) {
        uint64_t nonce_idx = (uint64_t)tid * NONCES_PER_THREAD + n;
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

        // Track best result
        if (matching_bits > local_best_match ||
            (matching_bits == local_best_match && meets_difficulty)) {
            local_best_match = matching_bits;
            local_best_nonce = nonce;
            local_best_score = calculate_difficulty_score(hash, d_job.target_hash);
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                local_best_hash[i] = hash[i];
            }
        }
    }

    // Warp-level reduction to find best result
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;

    // Use warp shuffle to find best in warp
    uint32_t warp_best_match = local_best_match;
    uint32_t warp_best_score = local_best_score;

    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        uint32_t other_match = __shfl_xor_sync(0xFFFFFFFF, warp_best_match, offset);
        uint32_t other_score = __shfl_xor_sync(0xFFFFFFFF, warp_best_score, offset);

        if (other_match > warp_best_match ||
            (other_match == warp_best_match && other_score > warp_best_score)) {
            warp_best_match = other_match;
            warp_best_score = other_score;
        }
    }

    // Lane 0 writes warp's best to shared memory
    if (lane_id == 0) {
        s_best_matches[warp_id] = warp_best_match;
        s_best_scores[warp_id] = warp_best_score;
    }
    __syncthreads();

    // Find thread with best result in warp
    bool i_have_best = (local_best_match == warp_best_match &&
                        local_best_score == warp_best_score);
    uint32_t best_lane = __ballot_sync(0xFFFFFFFF, i_have_best) & ((1 << lane_id) - 1);
    best_lane = __popc(best_lane);

    // Store results that meet difficulty threshold
    if (i_have_best && local_best_match >= d_job.difficulty) {
        uint32_t result_idx = atomicAdd(pool.count, 1);

        if (result_idx < pool.capacity) {
            MiningResult& result = pool.results[result_idx];
            result.nonce = local_best_nonce;
            result.matching_bits = local_best_match;
            result.difficulty_score = local_best_score;

            #pragma unroll
            for (int i = 0; i < 5; i++) {
                result.hash[i] = local_best_hash[i];
            }
        }
    }
}

// Host-side kernel launcher
extern "C" void launch_mining_kernel(
    const MiningJob& job,
    ResultPool& pool,
    const KernelConfig& config
) {
    // Upload job to constant memory
    cudaMemcpyToSymbolAsync(d_job, &job, sizeof(MiningJob), 0,
                            cudaMemcpyHostToDevice, config.stream);

    // Calculate work distribution
    uint64_t total_nonces = (uint64_t)config.blocks * config.threads_per_block * NONCES_PER_THREAD;

    // Launch kernel
    sha1_near_collision_kernel<<<config.blocks, config.threads_per_block,
                                 config.shared_memory_size, config.stream>>>(
        pool, job.nonce_offset, total_nonces
    );
}