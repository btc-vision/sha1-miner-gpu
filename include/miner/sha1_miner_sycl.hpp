#pragma once
#include <stdint.h>

//#include <iomanip>
//#include <sstream>
#include <string>
//#include <vector>

#include "sha1_miner.cuh"

//#include "gpu_platform.hpp"

// Only include OpenSSL for host code
#if !defined(__SYCL_DEVICE_ONLY__)
    #include <openssl/sha.h>
#else
    // Define SHA_DIGEST_LENGTH for device code if not already defined
    #ifndef SHA_DIGEST_LENGTH
        #define SHA_DIGEST_LENGTH 20
    #endif
#endif

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
