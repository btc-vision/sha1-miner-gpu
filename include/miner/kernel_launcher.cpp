// kernel_launcher.cpp - Vendor-specific kernel launcher
#include "sha1_miner.cuh"
#include "gpu_platform.hpp"
#include "../../src/mining_system.hpp"

// Forward declarations for vendor-specific kernels
#ifdef USE_HIP
extern void launch_mining_kernel_amd(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
);
#else
extern void launch_mining_kernel_nvidia(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
);
#endif

/**
 * Unified kernel launcher that dispatches to the appropriate vendor-specific kernel
 */
void launch_mining_kernel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config
) {
#ifdef USE_HIP
    // Use AMD HIP kernel
    launch_mining_kernel_amd(device_job, difficulty, nonce_offset, pool, config);
#else
    // Use NVIDIA CUDA kernel
    launch_mining_kernel_nvidia(device_job, difficulty, nonce_offset, pool, config);
#endif
}
