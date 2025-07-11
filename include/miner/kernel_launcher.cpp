#include "sha1_miner.cuh"

// Forward declarations of kernel launch functions
#ifdef USE_HIP
extern "C" void launch_mining_kernel_amd(
    const DeviceMiningJob& device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool& pool,
    const KernelConfig& config,
    uint64_t job_version
);
#else
extern void launch_mining_kernel_nvidia(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
);
#endif

// Unified kernel launch function
void launch_mining_kernel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
) {
#ifdef DEBUG_SHA1
    printf("[DEBUG] Launching mining kernel with difficulty=%u, nonce_offset=%llu\n",
           difficulty, nonce_offset);
#endif

#ifdef USE_HIP
    launch_mining_kernel_amd(device_job, difficulty, nonce_offset, pool, config, job_version);
#else
    launch_mining_kernel_nvidia(device_job, difficulty, nonce_offset, pool, config, job_version);
#endif
}
