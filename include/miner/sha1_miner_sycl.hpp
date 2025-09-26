#pragma once
#include <stdint.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_platform.hpp"

// Only include OpenSSL for host code
#if !defined(__SYCL_DEVICE_ONLY__)
    #include <openssl/sha.h>
#else
    // Define SHA_DIGEST_LENGTH for device code if not already defined
    #ifndef SHA_DIGEST_LENGTH
        #define SHA_DIGEST_LENGTH 20
    #endif
#endif

// SHA-1 constants
#define SHA1_BLOCK_SIZE          64
#define SHA1_DIGEST_SIZE         20
#define SHA1_ROUNDS              80
#define MAX_CANDIDATES_PER_BATCH 1024

#ifdef USE_SYCL
    #define NONCES_PER_THREAD         4096
    #define DEFAULT_THREADS_PER_BLOCK 256
#else
    #define NONCES_PER_THREAD         16384
    #define DEFAULT_THREADS_PER_BLOCK 256
#endif

// No CUDA __constant__ declarations for SYCL - these are handled by USM pointers

struct alignas(256) MiningJob
{
    uint8_t base_message[32];  // Base message to modify
    uint32_t target_hash[5];   // Target hash we're trying to match
    uint32_t difficulty;       // Number of bits that must match
    uint64_t nonce_offset;     // Starting nonce for this job
};

// Result structure for found candidates
struct alignas(16) MiningResult
{
    uint64_t nonce;
    uint32_t hash[5];
    uint32_t matching_bits;
    uint64_t job_version;
};

// Device-side mining job structure
struct alignas(64) DeviceMiningJob
{
    uint32_t base_message[8];  // Base message words
    uint32_t target_hash[5];   // Target hash
};

// Configuration structure for kernel execution
struct KernelConfig
{
    int blocks;
    int threads_per_block;
    int shared_memory_size;
};

// Result pool structure for managing mining results
struct ResultPool
{
    MiningResult *results;
    uint32_t *count;
    uint32_t capacity;
};

// Platform abstraction interface
class GPUMiner
{
public:
    virtual ~GPUMiner()                                                                                 = default;
    virtual bool initialize()                                                                           = 0;
    virtual void cleanup()                                                                              = 0;
    virtual void setBaseMessage(const uint32_t *base_msg_words)                                         = 0;
    virtual void launchKernel(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                              const ResultPool &pool, const KernelConfig &config, uint64_t job_version) = 0;
    virtual std::string getPlatformName() const                                                         = 0;
};

// Function declarations
void update_base_message_sycl(const uint32_t *base_msg_words);
bool initialize_sycl_runtime();
extern "C" void launch_mining_kernel_intel(const DeviceMiningJob &device_job, uint32_t difficulty,
                                           uint64_t nonce_offset, const ResultPool &pool, const KernelConfig &config,
                                           uint64_t job_version);
void cleanup_sycl_runtime();

// Utility functions
std::string format_hash_hex(const uint32_t hash[5]);
uint32_t count_leading_zeros_160bit(const uint32_t hash[5], const uint32_t target[5]);
void print_bits(const uint32_t hash[5], int count);
bool verify_sha1_hash(const uint8_t *message, size_t length, const uint32_t expected[5]);
void sha1_hash_bytes(const uint8_t *input, size_t length, uint32_t output[5]);
