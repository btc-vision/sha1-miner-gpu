#include "job_upload_api.h"
#include "sha1_miner.cuh"
#include <cuda_runtime.h>

// Global job storage for legacy compatibility
static MiningJob g_current_job;

extern "C" void upload_new_job(const uint8_t msg32[32], const uint32_t digest[5]) {
    // Copy message
    for (int i = 0; i < 32; i++) {
        g_current_job.base_message[i] = msg32[i];
    }

    // Copy target hash
    for (int i = 0; i < 5; i++) {
        g_current_job.target_hash[i] = digest[i];
    }

    // Set default difficulty (can be overridden)
    g_current_job.difficulty = 100;
    g_current_job.nonce_offset = 0;

    // Upload to GPU constant memory (handled by kernel launcher)
    // This is now done in launch_mining_kernel() for better stream management
}