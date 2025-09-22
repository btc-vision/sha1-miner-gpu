#include "sha1_miner.cuh"

#ifdef USE_SYCL
extern "C" void launch_mining_kernel_intel(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                           const ResultPool &pool, const KernelConfig &config, uint64_t job_version);
#elif USE_HIP
extern "C" void launch_mining_kernel_amd(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                         const ResultPool &pool, const KernelConfig &config, uint64_t job_version);
#else
extern void launch_mining_kernel_nvidia(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                                        const ResultPool &pool, const KernelConfig &config, uint64_t job_version);
#endif

// Unified kernel launch function
void launch_mining_kernel(const DeviceMiningJob &device_job, uint32_t difficulty, uint64_t nonce_offset,
                          const ResultPool &pool, const KernelConfig &config, uint64_t job_version)
{
#ifdef USE_SYCL
    launch_mining_kernel_intel(device_job, difficulty, nonce_offset, pool, config, job_version);
#elif USE_HIP
    launch_mining_kernel_amd(device_job, difficulty, nonce_offset, pool, config, job_version);
#else
    launch_mining_kernel_nvidia(device_job, difficulty, nonce_offset, pool, config, job_version);
#endif
}
