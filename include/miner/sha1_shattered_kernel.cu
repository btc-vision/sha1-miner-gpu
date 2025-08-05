#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "sha1_miner.cuh"

#include "../logging/logger.hpp"
#include "../shattered/m.hpp"

// Required vectors
std::vector<q13sol_t> q13sols;
std::vector<q14sol_t> q14sols;

#include "../shattered/lib/sha1detail.hpp"
#include "../shattered/lib/types.hpp"

// Define before including tables
#define QOFF 4

namespace host {
#include "../shattered/nc2/tables_org.hpp"
    using namespace tbl_org;
}  // namespace host

// Mining state
struct ShatteredState
{
    // Precomputed base solutions for different nonce ranges
    std::vector<std::vector<q14sol_t>> precomputed_bases;
    std::mutex bases_mutex;

    // Target info
    uint32_t target_hash[5];
    uint32_t min_difficulty;
    uint64_t job_version;

    // Results collection
    std::mutex results_mutex;
    std::vector<MiningResult> pending_results;

    // Control
    std::atomic<bool> running{false};
    std::atomic<uint64_t> nonces_processed{0};

    // Thread management
    std::thread worker_thread;

    // Statistics
    std::atomic<uint64_t> q53_solutions{0};
    std::atomic<uint64_t> near_collisions{0};
} g_shattered_state;

// Implement required callbacks
void process_q13sol(const uint32_t m1[80], const uint32_t Q1[85])
{
    q13sol_t sol;
    for (int i = 0; i < 16; i++) {
        sol.m[i] = m1[i];
    }
    q13sols.push_back(sol);
}

void process_q14sol(const uint32_t m1[80], const uint32_t Q1[85])
{
    q14sol_t sol;
    for (int i = 0; i < 16; i++) {
        sol.m[i] = m1[i];
        if (i < 16)
            sol.Q[i] = Q1[QOFF + i + 1];
    }
    q14sols.push_back(sol);
}

void compute_sha1_hash(const uint32_t m[16], uint32_t hash[5])
{
    // Constants
    const uint32_t K[4] = {0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6};
    // Message expansion
    uint32_t W[80];
    for (int i = 0; i < 16; i++) {
        W[i] = m[i];
    }

    for (int i = 16; i < 80; i++) {
        W[i] = rotate_left(W[i - 3] ^ W[i - 8] ^ W[i - 14] ^ W[i - 16], 1);
    }

    // Initialize working variables
    uint32_t a = 0x67452301;
    uint32_t b = 0xEFCDAB89;
    uint32_t c = 0x98BADCFE;
    uint32_t d = 0x10325476;
    uint32_t e = 0xC3D2E1F0;

    // Round 1
    for (int t = 0; t < 20; t++) {
        uint32_t temp = rotate_left(a, 5) + ((b & c) | (~b & d)) + e + K[0] + W[t];
        e             = d;
        d             = c;
        c             = rotate_left(b, 30);
        b             = a;
        a             = temp;
    }

    // Round 2
    for (int t = 20; t < 40; t++) {
        uint32_t temp = rotate_left(a, 5) + (b ^ c ^ d) + e + K[1] + W[t];
        e             = d;
        d             = c;
        c             = rotate_left(b, 30);
        b             = a;
        a             = temp;
    }

    // Round 3
    for (int t = 40; t < 60; t++) {
        uint32_t temp = rotate_left(a, 5) + ((b & c) | (b & d) | (c & d)) + e + K[2] + W[t];
        e             = d;
        d             = c;
        c             = rotate_left(b, 30);
        b             = a;
        a             = temp;
    }

    // Round 4
    for (int t = 60; t < 80; t++) {
        uint32_t temp = rotate_left(a, 5) + (b ^ c ^ d) + e + K[3] + W[t];
        e             = d;
        d             = c;
        c             = rotate_left(b, 30);
        b             = a;
        a             = temp;
    }

    // Add to hash values
    hash[0] = a + 0x67452301;
    hash[1] = b + 0xEFCDAB89;
    hash[2] = c + 0x98BADCFE;
    hash[3] = d + 0x10325476;
    hash[4] = e + 0xC3D2E1F0;
}

// SHA-1 utilities
void sha1_step_bw(int t, uint32_t Q[85], const uint32_t m[80])
{
    using namespace hashclash;

    if (t >= 0 && t < 80) {
        uint32_t f, k;

        if (t < 20) {
            f = sha1_f1(Q[QOFF + t + 1], rotate_left(Q[QOFF + t + 2], 30), rotate_left(Q[QOFF + t + 3], 30));
            k = 0x5A827999;
        } else if (t < 40) {
            f = sha1_f2(Q[QOFF + t + 1], rotate_left(Q[QOFF + t + 2], 30), rotate_left(Q[QOFF + t + 3], 30));
            k = 0x6ED9EBA1;
        } else if (t < 60) {
            f = sha1_f3(Q[QOFF + t + 1], rotate_left(Q[QOFF + t + 2], 30), rotate_left(Q[QOFF + t + 3], 30));
            k = 0x8F1BBCDC;
        } else {
            f = sha1_f4(Q[QOFF + t + 1], rotate_left(Q[QOFF + t + 2], 30), rotate_left(Q[QOFF + t + 3], 30));
            k = 0xCA62C1D6;
        }

        Q[QOFF + t] = rotate_right(Q[QOFF + t + 4] - rotate_left(Q[QOFF + t + 3], 30) - f - k - m[t], 5);
    }
}

__host__ __device__ uint32_t count_matching_bits(const uint32_t hash[5], const uint32_t target[5])
{
    uint32_t total_bits = 0;
    for (int i = 0; i < 5; i++) {
        uint32_t xor_val = hash[i] ^ target[i];
        if (xor_val == 0) {
            total_bits += 32;
        } else {
#ifdef __CUDA_ARCH__
            total_bits += __clz(xor_val);
#else
            total_bits += __builtin_clz(xor_val);
#endif
            break;
        }
    }
    return total_bits;
}

// Custom Q53 processor
void process_q53_for_mining()
{
    using namespace host;
    uint32_t q53idx;
    while ((q53idx = Q53SOLBUF.getreadidx(Q53SOLCTL)) != 0xFFFFFFFF) {
        if (!g_shattered_state.running.load())
            break;
        g_shattered_state.q53_solutions++;
        // Extract solution
        uint32_t m1[80], m2[80];
        uint32_t Q1[85], Q2[85];
        // Get Q values
        Q1[QOFF + 49] = q53_solutions_buf.get<0>(q53idx);
        Q1[QOFF + 50] = q53_solutions_buf.get<1>(q53idx);
        Q1[QOFF + 51] = q53_solutions_buf.get<2>(q53idx);
        Q1[QOFF + 52] = q53_solutions_buf.get<3>(q53idx);
        Q1[QOFF + 53] = q53_solutions_buf.get<4>(q53idx);
        // Get message words m37-m52
        for (int i = 0; i < 16; i++) {
            m1[37 + i] = q53_solutions_buf.get<5 + i>(q53idx);
        }
        // Expand message
        sha1_me_generalised(m1, 37);
        // Backward computation to get full message
        for (int i = 52; i >= 0; --i) {
            sha1_step_bw(i, Q1, m1);
        }
        // Verify we got correct initial values
        bool valid = true;
        for (int i = -4; i <= 0; i++) {
            if (Q1[QOFF + i] != Qset1mask[QOFF + i]) {
                valid = false;
                LOG_WARN("SHATTERED", "Invalid backward computation at Q", i);
                break;
            }
        }
        if (!valid)
            continue;
        // Create differential message
        for (int i = 0; i < 80; i++) {
            m2[i] = m1[i] ^ DV_DW[i];
        }
        // NOW COMPUTE ACTUAL SHA-1 HASHES
        uint32_t hash1[5], hash2[5];
        compute_sha1_hash(m1, hash1);
        compute_sha1_hash(m2, hash2);
        // Check near-collisions with actual hashes
        uint32_t bits1 = count_matching_bits(hash1, g_shattered_state.target_hash);
        uint32_t bits2 = count_matching_bits(hash2, g_shattered_state.target_hash);
        if (bits1 >= g_shattered_state.min_difficulty || bits2 >= g_shattered_state.min_difficulty) {
            std::lock_guard<std::mutex> lock(g_shattered_state.results_mutex);
            if (bits1 >= g_shattered_state.min_difficulty) {
                MiningResult result;
                // Use first 8 bytes of message as nonce
                result.nonce = ((uint64_t)m1[0] << 32) | m1[1];
                memcpy(result.hash, hash1, 20);
                result.matching_bits    = bits1;
                result.difficulty_score = bits1;
                result.job_version      = g_shattered_state.job_version;
                g_shattered_state.pending_results.push_back(result);
                g_shattered_state.near_collisions++;
                LOG_INFO("SHATTERED", "Found near-collision M1 with ", bits1, " matching bits");
            }
            if (bits2 >= g_shattered_state.min_difficulty) {
                MiningResult result;
                result.nonce = ((uint64_t)m2[0] << 32) | m2[1];
                memcpy(result.hash, hash2, 20);
                result.matching_bits    = bits2;
                result.difficulty_score = bits2;
                result.job_version      = g_shattered_state.job_version;
                g_shattered_state.pending_results.push_back(result);
                g_shattered_state.near_collisions++;

                LOG_INFO("SHATTERED", "Found near-collision M2 with ", bits2, " matching bits");
            }
        }

        g_shattered_state.nonces_processed += 2;
    }
}

// Precompute base solutions
void precompute_base_solutions(const uint8_t *base_message, uint64_t nonce_start)
{
    const int BASES_PER_NONCE_RANGE = 100;
    const uint64_t NONCE_RANGE_SIZE = 1000000;

    std::lock_guard<std::mutex> lock(g_shattered_state.bases_mutex);

    // Check if we already have bases for this nonce range
    size_t range_idx = nonce_start / NONCE_RANGE_SIZE;

    while (g_shattered_state.precomputed_bases.size() <= range_idx) {
        g_shattered_state.precomputed_bases.emplace_back();
    }

    if (!g_shattered_state.precomputed_bases[range_idx].empty()) {
        // Already computed for this range
        return;
    }

    LOG_INFO("SHATTERED", "Precomputing base solutions for nonce range ", range_idx);

    // Generate base solutions using nonce-based variations
    q13sols.clear();
    q14sols.clear();

    for (int i = 0; i < BASES_PER_NONCE_RANGE; i++) {
        q13sol_t q13sol;

        // Use nonce to create deterministic variations
        uint64_t variation_nonce = nonce_start + i * 1000;

        // Initialize from base message
        memcpy(q13sol.m, base_message, 32);

        // Apply nonce-based variations
        q13sol.m[0] ^= (variation_nonce >> 32) & 0xFFFF;
        q13sol.m[1] ^= (variation_nonce & 0xFFFFFFFF) & 0xFFFF;
        q13sol.m[2] ^= ((variation_nonce >> 16) & 0xFFFF);
        q13sol.m[3] ^= ((variation_nonce >> 48) & 0xFFFF);

        // Small variations in other words
        for (int j = 4; j < 16; j++) {
            q13sol.m[j] ^= ((variation_nonce >> (j * 4)) & 0xFF);
        }

        // Verify and expand
        if (verify(q13sol)) {
            step13nb(q13sol);
        }
    }

    // Store the computed bases
    g_shattered_state.precomputed_bases[range_idx] = q14sols;

    LOG_INFO("SHATTERED", "Generated ", q14sols.size(), " base solutions for range ", range_idx);
}

// Worker thread function
void shattered_worker_thread(uint64_t nonce_offset, const KernelConfig &config)
{
    // Get precomputed bases for this nonce range
    size_t range_idx = nonce_offset / 1000000;

    std::vector<q14sol_t> bases;
    {
        std::lock_guard<std::mutex> lock(g_shattered_state.bases_mutex);
        if (range_idx < g_shattered_state.precomputed_bases.size()) {
            bases = g_shattered_state.precomputed_bases[range_idx];
        }
    }

    if (bases.empty()) {
        LOG_ERROR("SHATTERED", "No base solutions for nonce range ", range_idx);
        return;
    }

    // Override process function
    void (*original_process)() = process_q53_solutions;
    process_q53_solutions      = process_q53_for_mining;

    try {
        // Run SHAttered attack with the provided kernel config
        cuda_main_with_config(bases, config);
    } catch (const std::exception &e) {
        LOG_ERROR("SHATTERED", "CUDA error: ", e.what());
    }

    // Restore original
    process_q53_solutions = original_process;
}

// Main kernel launch function - no CUDA initialization here
void launch_shattered_mining_kernel(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                    const ResultPool &pool, const KernelConfig &config, uint64_t job_version)
{
    // Validate configuration
    if (!pool.results || !pool.count || !pool.nonces_processed) {
        LOG_ERROR("SHATTERED", "Invalid pool pointers");
        return;
    }

    // Reset result count
    cudaError_t err = cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), config.stream);
    if (err != cudaSuccess) {
        LOG_ERROR("SHATTERED", "Failed to reset result count: ", cudaGetErrorString(err));
        return;
    }

    // Get target hash from device
    uint8_t base_message[32];
    cudaMemcpy(base_message, device_job.base_message, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_shattered_state.target_hash, device_job.target_hash, 20, cudaMemcpyDeviceToHost);

    g_shattered_state.min_difficulty = difficulty;
    g_shattered_state.job_version    = job_version;

    // Stop any existing work
    if (g_shattered_state.running.load()) {
        g_shattered_state.running = false;
        if (g_shattered_state.worker_thread.joinable()) {
            g_shattered_state.worker_thread.join();
        }
    }

    // Clear pending results
    {
        std::lock_guard<std::mutex> lock(g_shattered_state.results_mutex);
        g_shattered_state.pending_results.clear();
    }

    // Precompute base solutions if needed
    precompute_base_solutions(base_message, nonce_offset);

    LOG_DEBUG("SHATTERED", "Starting with difficulty ", difficulty, " bits, nonce offset ", nonce_offset);

    // Start worker thread
    g_shattered_state.running          = true;
    g_shattered_state.nonces_processed = 0;
    g_shattered_state.worker_thread    = std::thread(shattered_worker_thread, nonce_offset, config);

    // Run for a limited time (should be configurable)
    auto start_time   = std::chrono::steady_clock::now();
    auto max_duration = std::chrono::seconds(10);

    while (g_shattered_state.running.load() && std::chrono::steady_clock::now() - start_time < max_duration) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check if we have enough results
        {
            std::lock_guard<std::mutex> lock(g_shattered_state.results_mutex);
            if (g_shattered_state.pending_results.size() >= pool.capacity) {
                break;
            }
        }
    }

    // Stop worker
    g_shattered_state.running = false;
    if (g_shattered_state.worker_thread.joinable()) {
        g_shattered_state.worker_thread.join();
    }

    // Copy results back to GPU
    {
        std::lock_guard<std::mutex> lock(g_shattered_state.results_mutex);

        uint32_t count = std::min((uint32_t)g_shattered_state.pending_results.size(), pool.capacity);

        if (count > 0) {
            err = cudaMemcpyAsync(pool.results, g_shattered_state.pending_results.data(), count * sizeof(MiningResult),
                                  cudaMemcpyHostToDevice, config.stream);
            if (err != cudaSuccess) {
                LOG_ERROR("SHATTERED", "Failed to copy results: ", cudaGetErrorString(err));
                return;
            }

            err = cudaMemcpyAsync(pool.count, &count, sizeof(uint32_t), cudaMemcpyHostToDevice, config.stream);
            if (err != cudaSuccess) {
                LOG_ERROR("SHATTERED", "Failed to copy count: ", cudaGetErrorString(err));
                return;
            }
        }
    }

    // Update nonces processed
    uint64_t nonces = g_shattered_state.nonces_processed.load();
    err = cudaMemcpyAsync(pool.nonces_processed, &nonces, sizeof(uint64_t), cudaMemcpyHostToDevice, config.stream);
    if (err != cudaSuccess) {
        LOG_ERROR("SHATTERED", "Failed to copy nonces: ", cudaGetErrorString(err));
    }

    // Synchronize
    if (config.stream) {
        cudaStreamSynchronize(config.stream);
    }

    LOG_DEBUG("SHATTERED", "Completed - Q53 solutions: ", g_shattered_state.q53_solutions.load(),
              ", Near-collisions: ", g_shattered_state.near_collisions.load());
}
