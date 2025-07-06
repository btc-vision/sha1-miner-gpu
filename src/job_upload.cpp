#include <cuda_runtime.h>
#include "job_constants.cuh"
#include "job_upload_api.h"
#include "sha1_miner_core.cuh"

extern "C" void upload_new_job(const uint8_t msg32[32], const uint32_t digest[5]) {
    // Upload to original constant memory locations
    cudaMemcpyToSymbol(g_job_msg, msg32, 32, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_target, digest, 5 * 4, 0, cudaMemcpyHostToDevice);

    // Also update the MiningJob structure if needed
    MiningJob job;
    memcpy(job.base_msg, msg32, 32);
    memcpy(job.target_hash, digest, 20);

    // Default values for difficulty and nonce range (can be updated later)
    job.difficulty_bits = 50;  // Default to 50-bit near-collisions
    job.nonce_start = 0;
    job.nonce_range = 1ull << 32;

    cudaMemcpyToSymbol(g_job, &job, sizeof(MiningJob), 0, cudaMemcpyHostToDevice);
}