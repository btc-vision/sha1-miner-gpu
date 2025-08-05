#pragma once

#include <cuda_runtime.h>

#include "sha1_miner.cuh"

// Constants from SHAttered attack
#define SHATTERED_Q_OFFSET  4
#define SHATTERED_ROUNDS    80
#define SHATTERED_MSG_WORDS 16

// Neutral bit masks for fixed-message attack
struct ShatteredNeutralBits
{
    uint32_t w10_mask;  // Neutral bits for word 10
    uint32_t w11_mask;  // Neutral bits for word 11
    uint32_t w12_mask;  // Neutral bits for word 12
    uint32_t w13_mask;  // Neutral bits for word 13
    uint32_t w14_mask;  // Neutral bits for word 14
    uint32_t w15_mask;  // Neutral bits for word 15
};

// Intermediate state for differential attack
struct ShatteredState
{
    uint32_t Q[85];      // Q[-4] to Q[80]
    uint32_t W[80];      // Message expansion
    uint32_t delta[16];  // Message differential
};

// Result structure for shattered mining
struct ShatteredResult
{
    uint64_t nonce;
    uint32_t hash[5];
    uint32_t matching_bits;
    uint32_t delta[16];  // The differential that achieved this
    bool is_shattered;   // Flag to indicate this came from shattered method
};

// Launch the shattered mining kernel
void launch_shattered_mining_kernel(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                    const ResultPool &pool, const KernelConfig &config, uint64_t job_version);

// Initialize constants (call from host)
void init_shattered_constants();
