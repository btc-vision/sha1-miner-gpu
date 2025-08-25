#include <cuda_runtime.h>

#include <cstdio>

#include "sha1_miner.cuh"

// SHA-1 constants
#define K0 0x5A827999
#define K1 0x6ED9EBA1
#define K2 0x8F1BBCDC
#define K3 0xCA62C1D6

#define H0_0 0x67452301
#define H0_1 0xEFCDAB89
#define H0_2 0x98BADCFE
#define H0_3 0x10325476
#define H0_4 0xC3D2E1F0

// Optimized byte swap using PTX
__device__ __forceinline__ uint32_t bswap32_ptx(uint32_t x)
{
    uint32_t result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(result) : "r"(x));
    return result;
}

// Optimized count leading zeros - fully unrolled
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

// SHA-1 round macros for maximum performance
#define SHA1_ROUND_0_19(a, b, c, d, e, W_val)                                                                          \
    do {                                                                                                               \
        uint32_t f    = (b & c) | (~b & d);                                                                            \
        uint32_t temp = __funnelshift_l(a, a, 5) + f + e + K0 + W_val;                                                 \
        e             = d;                                                                                             \
        d             = c;                                                                                             \
        c             = __funnelshift_l(b, b, 30);                                                                     \
        b             = a;                                                                                             \
        a             = temp;                                                                                          \
    } while (0)

#define SHA1_ROUND_20_39(a, b, c, d, e, W_val)                                                                         \
    do {                                                                                                               \
        uint32_t f    = b ^ c ^ d;                                                                                     \
        uint32_t temp = __funnelshift_l(a, a, 5) + f + e + K1 + W_val;                                                 \
        e             = d;                                                                                             \
        d             = c;                                                                                             \
        c             = __funnelshift_l(b, b, 30);                                                                     \
        b             = a;                                                                                             \
        a             = temp;                                                                                          \
    } while (0)

#define SHA1_ROUND_40_59(a, b, c, d, e, W_val)                                                                         \
    do {                                                                                                               \
        uint32_t f    = (b & c) | (d & (b ^ c));                                                                       \
        uint32_t temp = __funnelshift_l(a, a, 5) + f + e + K2 + W_val;                                                 \
        e             = d;                                                                                             \
        d             = c;                                                                                             \
        c             = __funnelshift_l(b, b, 30);                                                                     \
        b             = a;                                                                                             \
        a             = temp;                                                                                          \
    } while (0)

#define SHA1_ROUND_60_79(a, b, c, d, e, W_val)                                                                         \
    do {                                                                                                               \
        uint32_t f    = b ^ c ^ d;                                                                                     \
        uint32_t temp = __funnelshift_l(a, a, 5) + f + e + K3 + W_val;                                                 \
        e             = d;                                                                                             \
        d             = c;                                                                                             \
        c             = __funnelshift_l(b, b, 30);                                                                     \
        b             = a;                                                                                             \
        a             = temp;                                                                                          \
    } while (0)

#define COMPUTE_W(t)                                                                                                   \
    W[t & 15] = __funnelshift_l(W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15],               \
                                W[(t - 3) & 15] ^ W[(t - 8) & 15] ^ W[(t - 14) & 15] ^ W[(t - 16) & 15], 1)

__constant__ uint32_t d_base_message[8];

__global__ void sha1_mining_kernel_nvidia(const uint32_t *__restrict__ target_hash, uint32_t difficulty,
                                          MiningResult *__restrict__ results, uint32_t *__restrict__ result_count,
                                          uint32_t result_capacity, uint64_t nonce_base, uint32_t nonces_per_thread,
                                          uint64_t job_version)
{
    const uint32_t tid               = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

    // Load target hash into registers
    uint32_t target[5];
#pragma unroll
    for (int i = 0; i < 5; i++) {
        target[i] = target_hash[i];
    }

    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_base + i;
        if (nonce == 0)
            continue;

        // Create message with nonce
        uint32_t msg_words[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            msg_words[j] = d_base_message[j];
        }

        // Apply nonce
        msg_words[6] ^= bswap32_ptx(nonce >> 32);
        msg_words[7] ^= bswap32_ptx(nonce & 0xFFFFFFFF);

        // Convert to big-endian and prepare W array
        uint32_t W[16];

        // Unrolled byte swap
        W[0] = bswap32_ptx(msg_words[0]);
        W[1] = bswap32_ptx(msg_words[1]);
        W[2] = bswap32_ptx(msg_words[2]);
        W[3] = bswap32_ptx(msg_words[3]);
        W[4] = bswap32_ptx(msg_words[4]);
        W[5] = bswap32_ptx(msg_words[5]);
        W[6] = bswap32_ptx(msg_words[6]);
        W[7] = bswap32_ptx(msg_words[7]);

        // Apply padding
        W[8]  = 0x80000000;
        W[9]  = 0;
        W[10] = 0;
        W[11] = 0;
        W[12] = 0;
        W[13] = 0;
        W[14] = 0;
        W[15] = 0x00000100;

        // Initialize working variables
        uint32_t a = H0_0;
        uint32_t b = H0_1;
        uint32_t c = H0_2;
        uint32_t d = H0_3;
        uint32_t e = H0_4;

        // FULLY UNROLLED SHA-1 rounds for maximum speed
        // Rounds 0-15 (no message schedule needed)
        SHA1_ROUND_0_19(a, b, c, d, e, W[0]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[1]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[2]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[3]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[4]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[5]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[6]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[7]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[8]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[9]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[10]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[11]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[12]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[13]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[14]);
        SHA1_ROUND_0_19(a, b, c, d, e, W[15]);

        // Rounds 16-19 with message schedule
        COMPUTE_W(16);
        SHA1_ROUND_0_19(a, b, c, d, e, W[0]);
        COMPUTE_W(17);
        SHA1_ROUND_0_19(a, b, c, d, e, W[1]);
        COMPUTE_W(18);
        SHA1_ROUND_0_19(a, b, c, d, e, W[2]);
        COMPUTE_W(19);
        SHA1_ROUND_0_19(a, b, c, d, e, W[3]);

        // Rounds 20-39
        COMPUTE_W(20);
        SHA1_ROUND_20_39(a, b, c, d, e, W[4]);
        COMPUTE_W(21);
        SHA1_ROUND_20_39(a, b, c, d, e, W[5]);
        COMPUTE_W(22);
        SHA1_ROUND_20_39(a, b, c, d, e, W[6]);
        COMPUTE_W(23);
        SHA1_ROUND_20_39(a, b, c, d, e, W[7]);
        COMPUTE_W(24);
        SHA1_ROUND_20_39(a, b, c, d, e, W[8]);
        COMPUTE_W(25);
        SHA1_ROUND_20_39(a, b, c, d, e, W[9]);
        COMPUTE_W(26);
        SHA1_ROUND_20_39(a, b, c, d, e, W[10]);
        COMPUTE_W(27);
        SHA1_ROUND_20_39(a, b, c, d, e, W[11]);
        COMPUTE_W(28);
        SHA1_ROUND_20_39(a, b, c, d, e, W[12]);
        COMPUTE_W(29);
        SHA1_ROUND_20_39(a, b, c, d, e, W[13]);
        COMPUTE_W(30);
        SHA1_ROUND_20_39(a, b, c, d, e, W[14]);
        COMPUTE_W(31);
        SHA1_ROUND_20_39(a, b, c, d, e, W[15]);
        COMPUTE_W(32);
        SHA1_ROUND_20_39(a, b, c, d, e, W[0]);
        COMPUTE_W(33);
        SHA1_ROUND_20_39(a, b, c, d, e, W[1]);
        COMPUTE_W(34);
        SHA1_ROUND_20_39(a, b, c, d, e, W[2]);
        COMPUTE_W(35);
        SHA1_ROUND_20_39(a, b, c, d, e, W[3]);
        COMPUTE_W(36);
        SHA1_ROUND_20_39(a, b, c, d, e, W[4]);
        COMPUTE_W(37);
        SHA1_ROUND_20_39(a, b, c, d, e, W[5]);
        COMPUTE_W(38);
        SHA1_ROUND_20_39(a, b, c, d, e, W[6]);
        COMPUTE_W(39);
        SHA1_ROUND_20_39(a, b, c, d, e, W[7]);

        // Rounds 40-59
        COMPUTE_W(40);
        SHA1_ROUND_40_59(a, b, c, d, e, W[8]);
        COMPUTE_W(41);
        SHA1_ROUND_40_59(a, b, c, d, e, W[9]);
        COMPUTE_W(42);
        SHA1_ROUND_40_59(a, b, c, d, e, W[10]);
        COMPUTE_W(43);
        SHA1_ROUND_40_59(a, b, c, d, e, W[11]);
        COMPUTE_W(44);
        SHA1_ROUND_40_59(a, b, c, d, e, W[12]);
        COMPUTE_W(45);
        SHA1_ROUND_40_59(a, b, c, d, e, W[13]);
        COMPUTE_W(46);
        SHA1_ROUND_40_59(a, b, c, d, e, W[14]);
        COMPUTE_W(47);
        SHA1_ROUND_40_59(a, b, c, d, e, W[15]);
        COMPUTE_W(48);
        SHA1_ROUND_40_59(a, b, c, d, e, W[0]);
        COMPUTE_W(49);
        SHA1_ROUND_40_59(a, b, c, d, e, W[1]);
        COMPUTE_W(50);
        SHA1_ROUND_40_59(a, b, c, d, e, W[2]);
        COMPUTE_W(51);
        SHA1_ROUND_40_59(a, b, c, d, e, W[3]);
        COMPUTE_W(52);
        SHA1_ROUND_40_59(a, b, c, d, e, W[4]);
        COMPUTE_W(53);
        SHA1_ROUND_40_59(a, b, c, d, e, W[5]);
        COMPUTE_W(54);
        SHA1_ROUND_40_59(a, b, c, d, e, W[6]);
        COMPUTE_W(55);
        SHA1_ROUND_40_59(a, b, c, d, e, W[7]);
        COMPUTE_W(56);
        SHA1_ROUND_40_59(a, b, c, d, e, W[8]);
        COMPUTE_W(57);
        SHA1_ROUND_40_59(a, b, c, d, e, W[9]);
        COMPUTE_W(58);
        SHA1_ROUND_40_59(a, b, c, d, e, W[10]);
        COMPUTE_W(59);
        SHA1_ROUND_40_59(a, b, c, d, e, W[11]);

        // Rounds 60-79
        COMPUTE_W(60);
        SHA1_ROUND_60_79(a, b, c, d, e, W[12]);
        COMPUTE_W(61);
        SHA1_ROUND_60_79(a, b, c, d, e, W[13]);
        COMPUTE_W(62);
        SHA1_ROUND_60_79(a, b, c, d, e, W[14]);
        COMPUTE_W(63);
        SHA1_ROUND_60_79(a, b, c, d, e, W[15]);
        COMPUTE_W(64);
        SHA1_ROUND_60_79(a, b, c, d, e, W[0]);
        COMPUTE_W(65);
        SHA1_ROUND_60_79(a, b, c, d, e, W[1]);
        COMPUTE_W(66);
        SHA1_ROUND_60_79(a, b, c, d, e, W[2]);
        COMPUTE_W(67);
        SHA1_ROUND_60_79(a, b, c, d, e, W[3]);
        COMPUTE_W(68);
        SHA1_ROUND_60_79(a, b, c, d, e, W[4]);
        COMPUTE_W(69);
        SHA1_ROUND_60_79(a, b, c, d, e, W[5]);
        COMPUTE_W(70);
        SHA1_ROUND_60_79(a, b, c, d, e, W[6]);
        COMPUTE_W(71);
        SHA1_ROUND_60_79(a, b, c, d, e, W[7]);
        COMPUTE_W(72);
        SHA1_ROUND_60_79(a, b, c, d, e, W[8]);
        COMPUTE_W(73);
        SHA1_ROUND_60_79(a, b, c, d, e, W[9]);
        COMPUTE_W(74);
        SHA1_ROUND_60_79(a, b, c, d, e, W[10]);
        COMPUTE_W(75);
        SHA1_ROUND_60_79(a, b, c, d, e, W[11]);
        COMPUTE_W(76);
        SHA1_ROUND_60_79(a, b, c, d, e, W[12]);
        COMPUTE_W(77);
        SHA1_ROUND_60_79(a, b, c, d, e, W[13]);
        COMPUTE_W(78);
        SHA1_ROUND_60_79(a, b, c, d, e, W[14]);
        COMPUTE_W(79);
        SHA1_ROUND_60_79(a, b, c, d, e, W[15]);

        // Final hash values
        uint32_t hash[5];
        hash[0] = a + H0_0;
        hash[1] = b + H0_1;
        hash[2] = c + H0_2;
        hash[3] = d + H0_3;
        hash[4] = e + H0_4;

        // Check difficulty using optimized function
        uint32_t matching_bits = count_leading_zeros_160bit(hash, target);

        // After your hash computation, force the compiler to keep the values
        if (matching_bits >= difficulty) {
            uint32_t idx = atomicAdd(result_count, 1);
            if (idx < result_capacity) {
                results[idx].nonce         = nonce;
                results[idx].matching_bits = 0;  // matching_bits;

// Force all hash values to be stored
#pragma unroll
                for (int j = 0; j < 5; j++) {
                    results[idx].hash[j] = hash[j];
                }
            }
        } else {
            // Even on non-matches, force some observable side effect
            // This prevents the compiler from eliminating the hash computation
            if ((nonce & 0xFFFFF) == 0) {                // Every ~1M nonces
                atomicMax(result_count, hash[0] & 0x1);  // Minimal side effect
            }
        }
    }
}

// Launcher function
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

    // printf("Grid configuration: blocks=%d, threads=%d\n", config.blocks, config.threads_per_block);

    dim3 gridDim(config.threads_per_block, 1, 1);
    dim3 blockDim(config.blocks, 1, 1);

    sha1_mining_kernel_nvidia<<<gridDim, blockDim, 0, config.stream>>>(device_job.target_hash, difficulty, pool.results,
                                                                       pool.count, pool.capacity, nonce_offset,
                                                                       NONCES_PER_THREAD, job_version);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
