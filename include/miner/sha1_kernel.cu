#include <cuda_runtime.h>

#include <cstdio>

#include "sha1_miner.cuh"

// SHA-1 constants
__device__ __constant__ uint32_t K[4]  = {0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6};
__device__ __constant__ uint32_t H0[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

// Optimized byte swap using PTX
__device__ __forceinline__ uint32_t bswap32_ptx(uint32_t x)
{
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
}

// Count leading zeros with PTX
__device__ __forceinline__ uint32_t count_leading_zeros_160bit(const uint32_t hash[5], const uint32_t target[5])
{
    uint32_t xor_val;
    uint32_t clz;

    xor_val = hash[0] ^ target[0];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return clz;
    }
    xor_val = hash[1] ^ target[1];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 32 + clz;
    }

    xor_val = hash[2] ^ target[2];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 64 + clz;
    }

    xor_val = hash[3] ^ target[3];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 96 + clz;
    }

    xor_val = hash[4] ^ target[4];
    if (xor_val != 0) {
        asm("clz.b32 %0, %1;" : "=r"(clz) : "r"(xor_val));
        return 128 + clz;
    }

    return 160;
}

__global__ void sha1_mining_kernel_nvidia(const uint8_t *__restrict__ base_message,
                                          const uint32_t *__restrict__ target_hash, uint32_t difficulty,
                                          MiningResult *__restrict__ results, uint32_t *__restrict__ result_count,
                                          uint32_t result_capacity, uint64_t nonce_base, uint32_t nonces_per_thread,
                                          uint64_t job_version)
{
    const uint32_t tid               = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // Load base message using vectorized access
    uint8_t base_msg[32];
    auto *base_msg_vec           = reinterpret_cast<uint4 *>(base_msg);
    const auto *base_message_vec = reinterpret_cast<const uint4 *>(base_message);
    base_msg_vec[0]              = base_message_vec[0];
    base_msg_vec[1]              = base_message_vec[1];

    // Load target
    uint32_t target[5];
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = target_hash[i];
    }

    uint32_t nonces_processed = 0;

    // Main mining loop
    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0)
            continue;

        nonces_processed++;

        // Create message copy with vectorized ops
        uint8_t msg_bytes[32];
        auto *msg_bytes_vec = reinterpret_cast<uint4 *>(msg_bytes);
        msg_bytes_vec[0]    = base_msg_vec[0];
        msg_bytes_vec[1]    = base_msg_vec[1];

        // Apply nonce efficiently - THIS IS THE KEY FIX
        auto *msg_words = reinterpret_cast<uint32_t *>(msg_bytes);
        msg_words[6] ^= bswap32_ptx(nonce >> 32);
        msg_words[7] ^= bswap32_ptx(nonce & 0xFFFFFFFF);

        // Convert to big-endian words for SHA-1
        uint32_t W[16];
        const auto *msg_words_local = reinterpret_cast<uint32_t *>(msg_bytes);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = bswap32_ptx(msg_words_local[j]);
        }

        // Apply SHA-1 padding
        W[8] = 0x80000000;
#pragma unroll
        for (int j = 9; j < 15; j++) {
            W[j] = 0;
        }
        W[15] = 0x00000100;  // Message length (256 bits) in big-endian

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
                W[t & 15]     = __funnelshift_l(temp, temp, 1);
            }
            uint32_t f    = (b & c) | (~b & d);
            uint32_t temp = __funnelshift_l(a, a, 5) + f + e + K[0] + W[t & 15];
            e             = d;
            d             = c;
            c             = __funnelshift_l(b, b, 30);
            b             = a;
            a             = temp;
        }

// Rounds 20-39
#pragma unroll
        for (int t = 20; t < 40; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15]     = __funnelshift_l(temp, temp, 1);
            uint32_t f    = b ^ c ^ d;
            temp          = __funnelshift_l(a, a, 5) + f + e + K[1] + W[t & 15];
            e             = d;
            d             = c;
            c             = __funnelshift_l(b, b, 30);
            b             = a;
            a             = temp;
        }

// Rounds 40-59
#pragma unroll
        for (int t = 40; t < 60; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15]     = __funnelshift_l(temp, temp, 1);
            uint32_t f    = (b & c) | (d & (b ^ c));
            temp          = __funnelshift_l(a, a, 5) + f + e + K[2] + W[t & 15];
            e             = d;
            d             = c;
            c             = __funnelshift_l(b, b, 30);
            b             = a;
            a             = temp;
        }

// Rounds 60-79
#pragma unroll
        for (int t = 60; t < 80; t++) {
            uint32_t temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15]     = __funnelshift_l(temp, temp, 1);
            uint32_t f    = b ^ c ^ d;
            temp          = __funnelshift_l(a, a, 5) + f + e + K[3] + W[t & 15];
            e             = d;
            d             = c;
            c             = __funnelshift_l(b, b, 30);
            b             = a;
            a             = temp;
        }

        // Add initial hash values
        uint32_t hash[5];
        hash[0] = a + H0[0];
        hash[1] = b + H0[1];
        hash[2] = c + H0[2];
        hash[3] = d + H0[3];
        hash[4] = e + H0[4];

        // Check difficulty
        uint32_t matching_bits = count_leading_zeros_160bit(hash, target);

        if (matching_bits >= difficulty) {
            uint32_t idx = atomicAdd(result_count, 1);
            if (idx < result_capacity) {
                results[idx].nonce            = nonce;
                results[idx].matching_bits    = matching_bits;
                results[idx].difficulty_score = matching_bits;
                results[idx].job_version      = job_version;
#pragma unroll
                for (int j = 0; j < 5; j++) {
                    results[idx].hash[j] = hash[j];
                }
            }
        }
    }
}

// Launcher
void launch_mining_kernel_nvidia(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                 const ResultPool &pool, const KernelConfig &config, uint64_t job_version)
{
    // Validate configuration
    if (config.blocks <= 0 || config.threads_per_block <= 0) {
        fprintf(stderr, "Invalid kernel configuration: blocks=%d, threads=%d\n", config.blocks,
                config.threads_per_block);
        return;
    }

    // Validate pool pointers
    if (!pool.results || !pool.count) {
        fprintf(stderr, "ERROR: Invalid pool pointers - results=%p, count=%p\n", pool.results, pool.count);
        return;
    }

    // Reset result count
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

    sha1_mining_kernel_nvidia<<<gridDim, blockDim, 0, config.stream>>>(
        device_job.base_message, device_job.target_hash, difficulty, pool.results, pool.count, pool.capacity,
        nonce_offset, NONCES_PER_THREAD, job_version);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
