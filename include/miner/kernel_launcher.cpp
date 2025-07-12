#include "sha1_miner.cuh"

#ifdef USE_HIP
extern "C" void launch_mining_kernel_amd(
    const DeviceMiningJob& device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool& pool,
    const KernelConfig& config,
    uint64_t job_version,
    uint32_t stream_id,
    uint64_t *assigned_nonces,
    uint32_t *conflict_counter
);
#else
extern void launch_mining_kernel_nvidia(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version,
    uint32_t stream_id,
    uint64_t *assigned_nonces,
    uint32_t *conflict_counter
);
#endif

// Unified kernel launch function
void launch_mining_kernel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version,
    uint32_t stream_id,
    uint64_t *assigned_nonces,
    uint32_t *conflict_counter
) {
#ifdef DEBUG_SHA1
    printf("[DEBUG] Launching mining kernel with difficulty=%u, nonce_offset=%llu\n",
           difficulty, nonce_offset);
#endif

#ifdef USE_HIP
    launch_mining_kernel_amd(device_job, difficulty, nonce_offset, pool, config, job_version, stream_id, assigned_nonces, conflict_counter);
#else
    launch_mining_kernel_nvidia(device_job, difficulty, nonce_offset, pool, config, job_version, stream_id,
                                assigned_nonces, conflict_counter);
#endif
}
