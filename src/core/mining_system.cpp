#include "mining_system.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "sha1_miner.cuh"

#include "architecture/gpu_architecture.hpp"
#include "config.hpp"
#include "utilities.hpp"

// Global system instance
std::unique_ptr<MiningSystem> g_mining_system;

// BestResultTracker implementation
BestResultTracker::BestResultTracker() : best_bits_(0) {}

bool BestResultTracker::isNewBest(const uint32_t matching_bits)
{
    std::lock_guard lock(mutex_);
    if (matching_bits > best_bits_) {
        best_bits_ = matching_bits;
        return true;
    }
    return false;
}

uint32_t BestResultTracker::getBestBits() const
{
    std::lock_guard lock(mutex_);
    return best_bits_;
}

void BestResultTracker::reset()
{
    std::lock_guard lock(mutex_);
    best_bits_ = 0;
}

// TimingStats implementation
void MiningSystem::TimingStats::reset()
{
    kernel_launch_time_ms    = 0;
    kernel_execution_time_ms = 0;
    result_copy_time_ms      = 0;
    total_kernel_time_ms     = 0;
    kernel_count             = 0;
}

void MiningSystem::TimingStats::print() const
{
    if (kernel_count == 0)
        return;
    std::cout << "\n[TIMING STATS] After " << kernel_count << " kernels:\n";
    std::cout << "  Average kernel execution: " << (kernel_execution_time_ms / kernel_count) << " ms\n";
    std::cout << "  Average result copy: " << (result_copy_time_ms / kernel_count) << " ms\n";
    std::cout << "  Total accumulated time: " << total_kernel_time_ms << " ms\n";
}

// MiningSystem implementation
MiningSystem::MiningSystem(const Config &config)
    : config_(config), device_props_(), gpu_vendor_(GPUVendor::UNKNOWN), best_tracker_()
{}

MiningSystem::~MiningSystem()
{
    cleanup();
}

GPUVendor MiningSystem::detectGPUVendor() const
{
    std::string device_name = device_props_.name;
    // Convert to lowercase for comparison
    std::ranges::transform(device_name, device_name.begin(), ::tolower);
    // Check for NVIDIA GPUs
    if (device_name.find("nvidia") != std::string::npos || device_name.find("geforce") != std::string::npos ||
        device_name.find("quadro") != std::string::npos || device_name.find("tesla") != std::string::npos ||
        device_name.find("titan") != std::string::npos || device_name.find("rtx") != std::string::npos ||
        device_name.find("gtx") != std::string::npos) {
        return GPUVendor::NVIDIA;
    }
    // Check for AMD GPUs
    if (device_name.find("amd") != std::string::npos || device_name.find("radeon") != std::string::npos ||
        device_name.find("vega") != std::string::npos || device_name.find("polaris") != std::string::npos ||
        device_name.find("navi") != std::string::npos || device_name.find("rdna") != std::string::npos ||
        device_name.find("gfx") != std::string::npos) {
        return GPUVendor::AMD;
    }
    return GPUVendor::UNKNOWN;
}

MiningSystem::UserSpecifiedFlags MiningSystem::detectUserSpecifiedValues() const
{
    UserSpecifiedFlags flags;
    if (const bool has_user_config = (user_config_ != nullptr); !has_user_config) {
        return flags;
    }

    const auto *mining_config = static_cast<const MiningConfig *>(user_config_);
    flags.threads             = mining_config->user_specified.threads_per_block && config_.threads_per_block > 0;
    flags.streams             = mining_config->user_specified.num_streams && config_.num_streams > 0;
    flags.blocks              = mining_config->user_specified.blocks_per_stream && config_.blocks_per_stream > 0;
    flags.buffer              = mining_config->user_specified.result_buffer_size && config_.result_buffer_size > 0;

    // Log what was user-specified
    if (flags.streams || flags.threads || flags.blocks || flags.buffer) {
        LOG_INFO("AUTOTUNE", "Detected user-specified parameters:");
        if (flags.streams)
            LOG_INFO("AUTOTUNE", "  - Streams: ", config_.num_streams);
        if (flags.threads)
            LOG_INFO("AUTOTUNE", "  - Threads per block: ", config_.threads_per_block);
        if (flags.blocks)
            LOG_INFO("AUTOTUNE", "  - Blocks per stream: ", config_.blocks_per_stream);
        if (flags.buffer)
            LOG_INFO("AUTOTUNE", "  - Result buffer size: ", config_.result_buffer_size);
    }
    return flags;
}

MiningSystem::OptimalConfig MiningSystem::getAMDOptimalConfig()
{
    OptimalConfig config;
#ifdef USE_HIP
    AMDArchitecture arch = AMDGPUDetector::detectArchitecture(device_props_);
    detected_arch_       = arch;
    printAMDArchitectureInfo(device_props_);
    if (AMDGPUDetector::hasKnownIssues(arch, device_props_.name)) {
        LOG_WARN("AUTOTUNE", "This GPU may have compatibility issues.");
        LOG_WARN("AUTOTUNE", "Consider updating ROCm/drivers or using reduced settings.");
    }

    switch (arch) {
        case AMDArchitecture::CDNA1:
        case AMDArchitecture::CDNA2:
        case AMDArchitecture::CDNA3:
        case AMDArchitecture::RDNA4:
            config.blocks_per_sm = 4;
            config.threads       = 256;
            config.streams       = 4;
            config.buffer_size   = 512;
            break;
            e AMDArchitecture::RDNA3 : config.blocks_per_sm = 6;
            config.threads                                  = 256;
            config.streams                                  = 4;
            config.buffer_size                              = 256;
            break;
            e AMDArchitecture::RDNA2 : config.blocks_per_sm = 4;
            config.threads                                  = 256;
            config.streams                                  = 4;
            config.buffer_size                              = 256;
            break;
            e AMDArchitecture::RDNA1 : config.blocks_per_sm = 4;
            config.threads                                  = 256;
            config.streams                                  = 2;
            config.buffer_size                              = 128;
            break;
        e AMDArchitecture::GCN5 : case AMDArchitecture::GCN4:
        case AMDArchitecture::GCN3:
            config.blocks_per_sm = 4;
            config.threads       = 256;
            config.streams       = 2;
            config.buffer_size   = 128;
            break;
        ault:
            config.blocks_per_sm = 2;
            config.threads       = 128;
            config.streams       = 2;
            config.buffer_size   = 128;
            break;
    }
#else
    // Fallback for non-HIP builds
    config.threads       = 256;
    config.blocks_per_sm = 4;
    config.streams       = 2;
    config.buffer_size   = 256;
#endif

    return config;
}

MiningSystem::OptimalConfig MiningSystem::getNVIDIAOptimalConfig() const
{
    OptimalConfig config;
    if (device_props_.major >= 8) {
        // Ampere and newer (RTX 30xx, 40xx, A100, etc.)
        config.blocks_per_sm = 16;
        config.threads       = 256;
        config.streams       = 8;
        config.buffer_size   = 512;
    } else if (device_props_.major == 7) {
        if (device_props_.minor >= 5) {
            // Turing (RTX 20xx, T4)
            config.blocks_per_sm = 8;
            config.threads       = 256;
            config.streams       = 4;
            config.buffer_size   = 256;
        } else {
            // Volta (V100, Titan V)
            config.blocks_per_sm = 8;
            config.threads       = 256;
            config.streams       = 4;
            config.buffer_size   = 256;
        }
    } else if (device_props_.major == 6) {
        // Pascal (GTX 10xx, P100)
        config.blocks_per_sm = 8;
        config.threads       = 256;
        config.streams       = 4;
        config.buffer_size   = 256;
    } else {
        // Maxwell and older
        config.blocks_per_sm = 4;
        config.threads       = 128;
        config.streams       = 4;
        config.buffer_size   = 128;
    }
    return config;
}

MiningSystem::OptimalConfig MiningSystem::determineOptimalConfig()
{
    // Detect GPU vendor
    gpu_vendor_ = detectGPUVendor();
    LOG_INFO("AUTOTUNE", "Detected GPU vendor: ",
             gpu_vendor_ == GPUVendor::NVIDIA ? "NVIDIA"
             : gpu_vendor_ == GPUVendor::AMD  ? "AMD"
                                              : "Unknown");
    OptimalConfig config{};
    if (gpu_vendor_ == GPUVendor::AMD) {
        config = getAMDOptimalConfig();
    } else if (gpu_vendor_ == GPUVendor::NVIDIA) {
        config = getNVIDIAOptimalConfig();
    } else {
        // Unknown vendor - use conservative defaults
        config.blocks_per_sm = 2;
        config.threads       = 128;
        config.streams       = 2;
        config.buffer_size   = 128;
    }
    return config;
}

void MiningSystem::applyUserSpecifiedValues(const UserSpecifiedFlags &user_flags, const OptimalConfig &optimal)
{
    if (!user_flags.threads) {
        config_.threads_per_block = optimal.threads;
    } else {
        LOG_INFO("AUTOTUNE", "Keeping user-specified threads per block: ", config_.threads_per_block);
    }
    if (!user_flags.streams) {
        config_.num_streams = optimal.streams;
    } else {
        LOG_INFO("AUTOTUNE", "Keeping user-specified number of streams: ", config_.num_streams);
    }
    if (!user_flags.blocks) {
        config_.blocks_per_stream = device_props_.multiProcessorCount * optimal.blocks_per_sm;
    } else {
        LOG_INFO("AUTOTUNE", "Keeping user-specified blocks per stream: ", config_.blocks_per_stream);
    }
    if (!user_flags.buffer) {
        config_.result_buffer_size = optimal.buffer_size;
    } else {
        LOG_INFO("AUTOTUNE", "Keeping user-specified result buffer size: ", config_.result_buffer_size);
    }
    // CRITICAL: Ensure result_buffer_size is never 0
    if (config_.result_buffer_size == 0) {
        config_.result_buffer_size = optimal.buffer_size;
        LOG_WARN("AUTOTUNE", "Result buffer size was 0, setting to ", config_.result_buffer_size);
    }
}

void MiningSystem::validateConfiguration()
{
    // Validate threads per block
    if (config_.threads_per_block > device_props_.maxThreadsPerBlock) {
        LOG_WARN("AUTOTUNE", "Requested threads per block (", config_.threads_per_block, ") exceeds device maximum (",
                 device_props_.maxThreadsPerBlock, "). Capping.");
        config_.threads_per_block = device_props_.maxThreadsPerBlock;
    }
    // Ensure threads are multiple of warp/wavefront size
    const int warp_size = device_props_.warpSize;
    if (config_.threads_per_block % warp_size != 0) {
        int adjusted = (config_.threads_per_block / warp_size) * warp_size;
        if (adjusted == 0)
            adjusted = warp_size;
        LOG_WARN("AUTOTUNE", "Adjusting threads per block from ", config_.threads_per_block, " to ", adjusted,
                 " (must be multiple of warp size ", warp_size, ")");
        config_.threads_per_block = adjusted;
    }
    // Apply architecture-specific block limits
    if (gpu_vendor_ == GPUVendor::AMD) {
        int max_blocks_per_stream = 256;
#ifdef USE_HIP
        switch (detected_arch_) {
            case AMDArchitecture::RDNA4:
                max_blocks_per_stream = 512;
                break;
            case AMDArchitecture::RDNA3:
                max_blocks_per_stream = 384;
                break;
            case AMDArchitecture::RDNA2:
                max_blocks_per_stream = 256;
                break;
            default:
                max_blocks_per_stream = 128;
                break;
        }
#endif
        if (config_.blocks_per_stream > max_blocks_per_stream) {
            LOG_WARN("AUTOTUNE", "Capping blocks per stream from ", config_.blocks_per_stream, " to ",
                     max_blocks_per_stream, " for this architecture");
            config_.blocks_per_stream = max_blocks_per_stream;
        }
    } else if (gpu_vendor_ == GPUVendor::NVIDIA) {
        int max_blocks = 2048;
        if (config_.blocks_per_stream > max_blocks) {
            config_.blocks_per_stream = max_blocks;
        }
    }
    // Ensure minimum configuration
    if (config_.num_streams < 1)
        config_.num_streams = 1;
    if (config_.blocks_per_stream < 1)
        config_.blocks_per_stream = 1;
    if (config_.threads_per_block < warp_size)
        config_.threads_per_block = warp_size;
    if (config_.result_buffer_size < 64)
        config_.result_buffer_size = 64;
}

void MiningSystem::adjustForMemoryConstraints(const UserSpecifiedFlags &user_flags)
{
    size_t free_mem, total_mem;
    gpuMemGetInfo(&free_mem, &total_mem);
    const size_t result_buffer_mem    = sizeof(MiningResult) * config_.result_buffer_size;
    const size_t working_mem_estimate = config_.blocks_per_stream * config_.threads_per_block * 512;
    const size_t mem_per_stream       = result_buffer_mem + working_mem_estimate + 2 * 1024 * 1024;

    const size_t required_mem = static_cast<size_t>(config_.num_streams) * mem_per_stream;

    if (const auto available_mem_threshold = static_cast<size_t>(static_cast<double>(free_mem) * 0.8);
        required_mem > available_mem_threshold) {
        if (user_flags.streams) {
            LOG_WARN("AUTOTUNE", "Current configuration requires ", (required_mem / (1024.0 * 1024.0)), " MB but only ",
                     (free_mem / (1024.0 * 1024.0)), " MB available.");
            LOG_WARN("AUTOTUNE", "Mining may fail due to insufficient memory.");
        } else {
            int max_streams_by_memory = static_cast<int>(available_mem_threshold / mem_per_stream);
            if (max_streams_by_memory < 1) {
                max_streams_by_memory = 1;
            }

            if (config_.num_streams > max_streams_by_memory) {
                LOG_INFO("AUTOTUNE", "Reducing streams from ", config_.num_streams, " to ", max_streams_by_memory,
                         " due to memory constraints");
                config_.num_streams = max_streams_by_memory;
            }
        }
    }
    // Check total thread count
    uint64_t total_threads = static_cast<uint64_t>(config_.blocks_per_stream) *
                             static_cast<uint64_t>(config_.threads_per_block) *
                             static_cast<uint64_t>(config_.num_streams);

    if (constexpr uint64_t MAX_TOTAL_THREADS = 1000000; total_threads > MAX_TOTAL_THREADS) {
        LOG_WARN("AUTOTUNE", "Total thread count (", total_threads, ") exceeds recommended maximum.");
        if (user_flags.streams && user_flags.threads) {
            LOG_WARN("AUTOTUNE", "Consider reducing streams or threads per block.");
        } else {
            LOG_INFO("AUTOTUNE", "Adjusting configuration...");
            while (total_threads > MAX_TOTAL_THREADS && !user_flags.streams && config_.num_streams > 1) {
                config_.num_streams--;
                total_threads = static_cast<uint64_t>(config_.blocks_per_stream) *
                                static_cast<uint64_t>(config_.threads_per_block) *
                                static_cast<uint64_t>(config_.num_streams);
            }
            while (total_threads > MAX_TOTAL_THREADS && !user_flags.blocks && config_.blocks_per_stream > 32) {
                config_.blocks_per_stream = (config_.blocks_per_stream * 3) / 4;
                total_threads             = static_cast<uint64_t>(config_.blocks_per_stream) *
                                static_cast<uint64_t>(config_.threads_per_block) *
                                static_cast<uint64_t>(config_.num_streams);
            }
        }
    }
}

void MiningSystem::logFinalConfiguration(const UserSpecifiedFlags &user_flags)
{
    LOG_INFO("AUTOTUNE", "=====================================");
    LOG_INFO("AUTOTUNE", "Final Mining Configuration:");
    LOG_INFO("AUTOTUNE", "=====================================");
    LOG_INFO("AUTOTUNE", "GPU: ", device_props_.name);
    LOG_INFO("AUTOTUNE", "Compute Capability: ", device_props_.major, ".", device_props_.minor);
    LOG_INFO("AUTOTUNE", "SMs/CUs: ", device_props_.multiProcessorCount);
    if (gpu_vendor_ == GPUVendor::NVIDIA) {
        LOG_INFO("AUTOTUNE", "Blocks per SM: ", (config_.blocks_per_stream / device_props_.multiProcessorCount));
    } else if (gpu_vendor_ == GPUVendor::AMD) {
        LOG_INFO("AUTOTUNE", "Blocks per CU: ", (config_.blocks_per_stream / device_props_.multiProcessorCount));
#ifdef USE_HIP
        LOG_INFO("AUTOTUNE", "Architecture: ", AMDGPUDetector::getArchitectureName(detected_arch_));
#endif
    }
    LOG_INFO("AUTOTUNE", "-------------------------------------");
    LOG_INFO("AUTOTUNE", "Performance Settings:");
    LOG_INFO("AUTOTUNE", "  Streams: ", config_.num_streams,
             user_flags.streams ? " (user-specified)" : " (auto-tuned)");
    LOG_INFO("AUTOTUNE", "  Blocks per stream: ", config_.blocks_per_stream,
             user_flags.blocks ? " (user-specified)" : " (auto-tuned)");
    LOG_INFO("AUTOTUNE", "  Threads per block: ", config_.threads_per_block,
             user_flags.threads ? " (user-specified)" : " (auto-tuned)");
    LOG_INFO("AUTOTUNE", "  Result buffer size: ", config_.result_buffer_size,
             user_flags.buffer ? " (user-specified)" : " (auto-tuned)");
    // Calculate metrics
    uint64_t total_threads = static_cast<uint64_t>(config_.blocks_per_stream) *
                             static_cast<uint64_t>(config_.threads_per_block) *
                             static_cast<uint64_t>(config_.num_streams);
    int max_threads_per_sm = device_props_.maxThreadsPerMultiProcessor;
    int threads_per_sm = (config_.blocks_per_stream / device_props_.multiProcessorCount) * config_.threads_per_block;
    float occupancy    = (float)threads_per_sm / (float)max_threads_per_sm * 100.0f;
    LOG_INFO("AUTOTUNE", "-------------------------------------");
    LOG_INFO("AUTOTUNE", "Calculated Metrics:");
    LOG_INFO("AUTOTUNE", "  Total concurrent threads: ", total_threads);
    LOG_INFO("AUTOTUNE", "  Theoretical occupancy: ", std::fixed, std::setprecision(1), occupancy, "%");
    LOG_INFO("AUTOTUNE", "  Hashes per kernel: ", getHashesPerKernel(), " (~", std::setprecision(2),
             (getHashesPerKernel() / 1e9), " GH)");

    // Memory usage
    size_t free_mem, total_mem;
    gpuMemGetInfo(&free_mem, &total_mem);
    size_t result_buffer_mem    = sizeof(MiningResult) * config_.result_buffer_size;
    size_t working_mem_estimate = config_.blocks_per_stream * config_.threads_per_block * 512;
    size_t mem_per_stream       = result_buffer_mem + working_mem_estimate + (2 * 1024 * 1024);
    size_t total_mem_usage      = config_.num_streams * mem_per_stream;

    LOG_INFO("AUTOTUNE", "-------------------------------------");
    LOG_INFO("AUTOTUNE", "Memory Usage:");
    LOG_INFO("AUTOTUNE", "  Per stream: ", std::setprecision(2), (mem_per_stream / (1024.0 * 1024.0)), " MB");
    LOG_INFO("AUTOTUNE", "  Total estimated: ", (total_mem_usage / (1024.0 * 1024.0)), " MB");
    LOG_INFO("AUTOTUNE", "  Available GPU memory: ", (free_mem / (1024.0 * 1024.0)), " MB");
    LOG_INFO("AUTOTUNE", "=====================================");
}

void MiningSystem::autoTuneParameters()
{
    LOG_INFO("AUTOTUNE", "Starting auto-tune process...");
    // Step 1: Detect user-specified values
    UserSpecifiedFlags user_flags = detectUserSpecifiedValues();
    // Step 2: Determine optimal configuration based on GPU
    OptimalConfig optimal = determineOptimalConfig();
    // Step 3: Apply configuration with user preferences
    applyUserSpecifiedValues(user_flags, optimal);
    // Step 4: Validate and adjust configuration
    validateConfiguration();
    // Step 5: Adjust for memory constraints
    adjustForMemoryConstraints(user_flags);

    // Step 6: Final validation
    validateConfiguration();

    // Step 7: Log final configuration
    logFinalConfiguration(user_flags);
}

bool MiningSystem::initialize()
{
    std::lock_guard lock(system_mutex_);

    // First, check if any GPU is available
    int device_count = 0;
    gpuError_t err   = gpuGetDeviceCount(&device_count);
    if (err != gpuSuccess) {
        std::cerr << "Failed to get GPU device count: " << gpuGetErrorString(err) << "\n";
        std::cerr << "Is the GPU driver installed and running?\n";
        return false;
    }

    if (device_count == 0) {
        std::cerr << "No GPU devices found!\n";
        return false;
    }

    if (config_.device_id >= device_count) {
        std::cerr << "Invalid device ID " << config_.device_id << ". Available devices: 0-" << (device_count - 1)
                  << "\n";
        return false;
    }

    // Reset any previous errors
    gpuGetLastError();

    // Set device with error checking
    err = gpuSetDevice(config_.device_id);
    if (err != gpuSuccess) {
        std::cerr << "Failed to set GPU device " << config_.device_id << ": " << gpuGetErrorString(err) << "\n";
        // Try to provide more specific error information
#ifdef USE_HIP
        if (err == hipErrorInvalidDevice) {
            std::cerr << "Device " << config_.device_id << " is not a valid HIP device\n";
        } else if (err == hipErrorNoDevice) {
            std::cerr << "No HIP devices available\n";
        }
#else
        if (err == cudaErrorInvalidDevice) {
            std::cerr << "Device " << config_.device_id << " is not a valid CUDA device\n";
        } else if (err == cudaErrorNoDevice) {
            std::cerr << "No CUDA devices available\n";
        }
#endif
        return false;
    }

    // Verify we can communicate with the device
    err = gpuDeviceSynchronize();
    if (err != gpuSuccess) {
        std::cerr << "Failed to synchronize with device: " << gpuGetErrorString(err) << "\n";
        std::cerr << "The GPU may be in a bad state or the driver may need to be restarted\n";
        // Try to reset the device
        std::cerr << "Attempting device reset...\n";
#ifdef USE_HIP
        err = hipDeviceReset();
#else
        err = cudaDeviceReset();
#endif
        if (err != gpuSuccess) {
            std::cerr << "Device reset failed: " << gpuGetErrorString(err) << "\n";
            return false;
        }
        // Try setting device again after reset
        err = gpuSetDevice(config_.device_id);
        if (err != gpuSuccess) {
            std::cerr << "Failed to set device after reset: " << gpuGetErrorString(err) << "\n";
            return false;
        }
    }

    // Get device properties
    err = gpuGetDeviceProperties(&device_props_, config_.device_id);
    if (err != gpuSuccess) {
        std::cerr << "Failed to get device properties: " << gpuGetErrorString(err) << "\n";
        return false;
    }

    // Check if device is in prohibited mode (Windows TCC/WDDM issues)
#ifdef _WIN32
    if (device_props_.tccDriver) {
        std::cout << "Device is running in TCC mode\n";
    } else {
        std::cout << "Device is running in WDDM mode\n";
        // On Windows, WDDM mode has a timeout that can cause issues
        std::cout << "Note: WDDM mode has a 2-second timeout. Long-running kernels may fail.\n";
    }
#endif

    // Check compute capability
    if (device_props_.major < 3) {
        std::cerr << "GPU compute capability " << device_props_.major << "." << device_props_.minor
                  << " is too old. Minimum required: 3.0\n";
        return false;
    }

    // Print device info
    std::cout << "SHA-1 OP_NET Miner (" << getGPUPlatformName() << ")\n";
    std::cout << "=====================================\n";
    std::cout << "Device: " << device_props_.name << "\n";
    std::cout << "Compute Capability: " << device_props_.major << "." << device_props_.minor << "\n";
    std::cout << "SMs/CUs: " << device_props_.multiProcessorCount << "\n";
    std::cout << "Warp/Wavefront Size: " << device_props_.warpSize << "\n";
    std::cout << "Max Threads per Block: " << device_props_.maxThreadsPerBlock << "\n";
    std::cout << "Total Global Memory: " << (device_props_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    // Check available memory
    size_t free_mem, total_mem;
    err = gpuMemGetInfo(&free_mem, &total_mem);
    if (err == gpuSuccess) {
        std::cout << "Available Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";

        if (free_mem < 100 * 1024 * 1024) {
            // Less than 100MB free
            std::cerr << "WARNING: Very low GPU memory available. Mining may fail.\n";
        }
    }
    std::cout << "\n";

    autoTuneParameters();

    // Validate thread configuration
    if (config_.threads_per_block % device_props_.warpSize != 0 ||
        config_.threads_per_block > device_props_.maxThreadsPerBlock) {
        std::cerr << "Invalid thread configuration\n";
        std::cerr << "Threads per block must be multiple of " << device_props_.warpSize
                  << " and <= " << device_props_.maxThreadsPerBlock << "\n";
        return false;
    }

    // Initialize GPU resources
    if (!initializeGPUResources()) {
        std::cerr << "Failed to initialize GPU resources\n";
        return false;
    }

    // Set up L2 cache persistence for newer architectures
#ifdef USE_HIP
    // AMD doesn't have the same L2 persistence API
#else
    if (device_props_.major >= 8) {
        gpuDeviceSetLimit(gpuLimitPersistingL2CacheSize, device_props_.l2CacheSize);
    }
#endif

    start_time_ = std::chrono::steady_clock::now();
    timing_stats_.reset();

    std::cout << "Mining Configuration:\n";
    std::cout << "  Platform: " << getGPUPlatformName() << "\n";
    std::cout << "  Streams: " << config_.num_streams << "\n";
    std::cout << "  Blocks/Stream: " << config_.blocks_per_stream << "\n";
    std::cout << "  Threads/Block: " << config_.threads_per_block << "\n";
    std::cout << "  Total Threads: " << getTotalThreads() << "\n";
    std::cout << "  Hashes/Kernel: " << getHashesPerKernel() << " (~" << (getHashesPerKernel() / 1e9) << " GH)\n\n";

    return true;
}

uint64_t MiningSystem::runMiningLoopInterruptibleWithOffset(const MiningJob &job,
                                                            const std::function<bool()> &should_continue,
                                                            const uint64_t start_nonce)
{
    // Copy job to device
    for (int i = 0; i < config_.num_streams; i++) {
        device_jobs_[i].copyFromHost(job);
    }

    LOG_INFO("LOOP", "Reset mining system and prepare for new job");

    // Reset flags and counters
    stop_mining_ = false;
    best_tracker_.reset();
    total_hashes_ = 0;
    clearResults();

    // Initialize per-stream data
    std::vector<StreamData> stream_data(config_.num_streams);

    // Create events for each stream
    kernel_complete_events_.resize(config_.num_streams);
    kernel_launch_times_.resize(config_.num_streams);

    for (int i = 0; i < config_.num_streams; i++) {
        gpuEventCreateWithFlags(&kernel_complete_events_[i], gpuEventDisableTiming);
        stream_data[i].last_nonces_processed = 0;
        stream_data[i].busy                  = false;
        gpuMemsetAsync(gpu_pools_[i].nonces_processed, 0, sizeof(uint64_t), streams_[i]);
    }

    // Nonce distribution - START FROM PROVIDED OFFSET
    uint64_t nonce_stride        = getHashesPerKernel();
    uint64_t global_nonce_offset = start_nonce;

    LOG_INFO("MINING", "Starting mining from nonce offset: ", global_nonce_offset);

    // Launch initial kernels on all streams
    for (int i = 0; i < config_.num_streams; i++) {
        launchKernelOnStream(i, global_nonce_offset, job);
        stream_data[i].nonce_offset = global_nonce_offset;
        stream_data[i].busy         = true;
        global_nonce_offset += nonce_stride;
    }

    // Main mining loop
    while (!g_shutdown && !stop_mining_ && should_continue()) {
        // Check for completed kernels using events
        bool found_completed = false;
        int completed_stream = -1;

        for (int i = 0; i < config_.num_streams; i++) {
            if (!stream_data[i].busy)
                continue;

            gpuError_t status = gpuEventQuery(kernel_complete_events_[i]);
            if (status == gpuSuccess) {
                completed_stream = i;
                found_completed  = true;
                break;
            }
        }

        if (!found_completed) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        // Process the completed stream
        processStreamResults(completed_stream, stream_data[completed_stream]);
        stream_data[completed_stream].busy = false;

        // Check if we should stop before launching new work
        if (!should_continue()) {
            LOG_INFO("MINING", "Stopping work generation at nonce offset: ", global_nonce_offset);
            break;
        }

        // Launch new work on this stream
        launchKernelOnStream(completed_stream, global_nonce_offset, job);
        stream_data[completed_stream].nonce_offset = global_nonce_offset;
        stream_data[completed_stream].busy         = true;
        global_nonce_offset += nonce_stride;
    }

    // Wait for all remaining kernels to complete
    for (int i = 0; i < config_.num_streams; i++) {
        if (stream_data[i].busy) {
            gpuEventSynchronize(kernel_complete_events_[i]);
            processStreamResults(i, stream_data[i]);
        }
    }

    // Cleanup events
    for (int i = 0; i < config_.num_streams; i++) {
        gpuEventDestroy(kernel_complete_events_[i]);
    }

    LOG_DEBUG("MINING", "Mining stopped at nonce offset: ", global_nonce_offset);
    return global_nonce_offset;
}

// Simple wrapper for infinite mining
void MiningSystem::runMiningLoop(const MiningJob &job)
{
    std::cout << "Starting infinite mining...\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Target hash: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << job.target_hash[i] << " ";
    }
    std::cout << "\n" << std::dec;
    std::cout << "Only new best matches will be reported.\n";
    std::cout << "Press Ctrl+C to stop mining.\n";
    std::cout << "=====================================\n\n";

    // Reset job version to 0 for non-pool mining
    current_job_version_ = 0;

    // Just call the unified implementation with "always continue" lambda
    runMiningLoopInterruptibleWithOffset(job, []() { return !g_shutdown; }, 1);
}

// Update launchKernelOnStream to NOT modify global_nonce_offset
void MiningSystem::launchKernelOnStream(const int stream_idx, const uint64_t nonce_offset, const MiningJob &job)
{
    // Configure kernel
    KernelConfig config{};
    config.blocks             = config_.blocks_per_stream;
    config.threads_per_block  = config_.threads_per_block;
    config.stream             = streams_[stream_idx];
    config.shared_memory_size = 0;

    // Record launch time for performance tracking
    kernel_launch_times_[stream_idx] = std::chrono::high_resolution_clock::now();

    // Launch the mining kernel with current job version
    launch_mining_kernel(device_jobs_[stream_idx], job.difficulty,
                         nonce_offset,  // Use the offset directly
                         gpu_pools_[stream_idx], config, current_job_version_);

    // Record event when kernel completes
    gpuEventRecord(kernel_complete_events_[stream_idx], streams_[stream_idx]);

    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        LOG_ERROR("MINING", "Kernel launch failed on stream ", stream_idx, ": ", gpuGetErrorString(err));
        throw std::runtime_error("Kernel launch failed");
    }
}

void MiningSystem::processStreamResults(const int stream_idx, StreamData &stream_data)
{
    // Get actual nonces processed
    uint64_t actual_nonces = 0;
    gpuMemcpyAsync(&actual_nonces, gpu_pools_[stream_idx].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost,
                   streams_[stream_idx]);

    // Ensure the copy is complete
    gpuStreamSynchronize(streams_[stream_idx]);

    // Update hash count
    const uint64_t nonces_this_kernel = actual_nonces - stream_data.last_nonces_processed;
    total_hashes_ += nonces_this_kernel;
    stream_data.last_nonces_processed = actual_nonces;

    // Process mining results
    processResultsOptimized(stream_idx);

    // Update timing statistics
    {
        const auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::high_resolution_clock::now() - kernel_launch_times_[stream_idx])
                                     .count() /
                                 1000.0;
        std::lock_guard lock(timing_mutex_);
        timing_stats_.kernel_execution_time_ms += kernel_time;
        timing_stats_.kernel_count++;
    }
}

void MiningSystem::sync() const
{
    // Synchronize all streams in this MiningSystem instance
    for (int i = 0; i < config_.num_streams; i++) {
        if (streams_[i]) {
            gpuStreamSynchronize(streams_[i]);
        }
    }
}

bool MiningSystem::validateStreams()
{
    for (int i = 0; i < config_.num_streams; i++) {
        if (!streams_[i]) {
            LOG_ERROR("MINING", "Stream ", i, " is null");
            return false;
        }
        // Test stream by trying to record an event
        gpuEvent_t test_event;
        gpuError_t err = gpuEventCreate(&test_event);
        if (err != gpuSuccess) {
            LOG_ERROR("MINING", "Failed to create test event: ", gpuGetErrorString(err));
            return false;
        }

        err = gpuEventRecord(test_event, streams_[i]);
        if (err != gpuSuccess) {
            LOG_ERROR("MINING", "Stream ", i, " is invalid: ", gpuGetErrorString(err));
            gpuEventDestroy(test_event);
            return false;
        }

        // Clean up test event
        gpuEventDestroy(test_event);
    }
    return true;
}

void MiningSystem::updateJobLive(const MiningJob &job, uint64_t job_version)
{
    // Store current job version first
    current_job_version_ = job_version;

    // Update the device jobs with new data
    for (int i = 0; i < config_.num_streams; i++) {
        // Copy new job data to device
        gpuMemcpyAsync(device_jobs_[i].base_message, job.base_message, 32, gpuMemcpyHostToDevice, streams_[i]);
        gpuMemcpyAsync(device_jobs_[i].target_hash, job.target_hash, 5 * sizeof(uint32_t), gpuMemcpyHostToDevice,
                       streams_[i]);
    }

    // Synchronize all streams to ensure job update is complete
    for (int i = 0; i < config_.num_streams; i++) {
        gpuStreamSynchronize(streams_[i]);
    }

    LOG_INFO("MINING", "Job update to version ", job_version, " completed");
}

void MiningSystem::processResultsOptimized(int stream_idx)
{
    auto &pool    = gpu_pools_[stream_idx];
    auto &results = pinned_results_[stream_idx];

    // Get result count
    uint32_t count;
    gpuMemcpyAsync(&count, pool.count, sizeof(uint32_t), gpuMemcpyDeviceToHost, streams_[stream_idx]);
    gpuStreamSynchronize(streams_[stream_idx]);

    if (count == 0)
        return;

    // Limit to capacity
    if (count > pool.capacity) {
        LOG_WARN("MINING", "Result count (", count, ") exceeds capacity (", pool.capacity, "), capping results");
        count = pool.capacity;
    }

    // Copy results
    auto copy_start = std::chrono::high_resolution_clock::now();

    gpuMemcpyAsync(results, pool.results, sizeof(MiningResult) * count, gpuMemcpyDeviceToHost, streams_[stream_idx]);
    gpuStreamSynchronize(streams_[stream_idx]);

    auto copy_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - copy_start)
            .count() /
        1000.0;

    LOG_TRACE("MINING", "Copied ", count, " results from stream ", stream_idx, " in ", copy_time, " ms");

    // Update timing stats with proper lock
    {
        std::lock_guard lock(timing_mutex_);
        timing_stats_.result_copy_time_ms += copy_time;
    }

    // Process results
    std::vector<MiningResult> valid_results;
    uint32_t stale_count = 0;

    for (uint32_t i = 0; i < count; i++) {
        if (results[i].nonce == 0)
            continue;

        // Check if result is from current job version
        if (results[i].job_version != current_job_version_) {
            // Skip stale results from old job versions
            stale_count++;
            LOG_TRACE("MINING", "Skipping stale result from job version ", results[i].job_version);
            continue;
        }

        // Store all valid results from current job
        valid_results.push_back(results[i]);

        // Track best result
        if (best_tracker_.isNewBest(results[i].matching_bits)) {
            // Calculate elapsed time
            auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

            std::string hash_str = "0x";
            for (int j = 0; j < 5; j++) {
                char buf[9];
                snprintf(buf, sizeof(buf), "%08x", results[i].hash[j]);
                hash_str += buf;
            }

            double hash_rate = static_cast<double>(total_hashes_.load()) / elapsed.count() / 1e9;

            // Helper function to pad string to fixed width
            auto pad_right = [](const std::string &str, size_t width) -> std::string {
                if (str.length() >= width)
                    return str;
                return str + std::string(width - str.length(), ' ');
            };

            // Format all values
            std::string time_str     = std::to_string(elapsed.count()) + "s";
            std::string platform_str = getGPUPlatformName();

            // Format nonce using snprintf to avoid locale issues
            char nonce_buffer[32];
            snprintf(nonce_buffer, sizeof(nonce_buffer), "0x%llx", (unsigned long long)results[i].nonce);
            std::string nonce_str = nonce_buffer;

            std::string bits_str = std::to_string(results[i].matching_bits);

            std::stringstream rate_stream;
            rate_stream << std::fixed << std::setprecision(2) << hash_rate << " GH/s";
            std::string rate_str = rate_stream.str();

            // Build complete colored strings to avoid logger parsing issues
            std::stringstream line;

            // Log new best as a single line
            LOG_INFO("MINING", Color::BRIGHT_CYAN, "NEW BEST! ", Color::RESET, "Time: ", Color::BRIGHT_WHITE, time_str,
                     Color::RESET, " | Nonce: ", Color::BRIGHT_GREEN, nonce_str, Color::RESET,
                     " | Bits: ", Color::BRIGHT_MAGENTA, bits_str, Color::RESET, " | Hash: ", Color::BRIGHT_YELLOW,
                     hash_str);
        }

        ++total_candidates_;
    }

    if (stale_count > 0) {
        LOG_DEBUG("MINING", "Discarded ", stale_count, " stale results from stream ", stream_idx);
    }

    if (!valid_results.empty()) {
        LOG_DEBUG("MINING", "Found ", valid_results.size(), " valid results from stream ", stream_idx);
    }

    // Store results for batch processing
    if (!valid_results.empty()) {
        {
            std::lock_guard results_lock(all_results_mutex_);
            all_results_.insert(all_results_.end(), valid_results.begin(), valid_results.end());
        }

        // Call the callback if set
        {
            std::lock_guard callback_lock(callback_mutex_);
            if (result_callback_) {
                LOG_TRACE("MINING", "Invoking result callback with ", valid_results.size(), " results");
                result_callback_(valid_results);
            }
        }
    }

    // Reset pool count
    gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[stream_idx]);
    LOG_TRACE("MINING", "Reset result count for stream ", stream_idx);
}

MiningStats MiningSystem::getStats() const
{
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

    MiningStats stats{};
    stats.hashes_computed  = total_hashes_.load();
    stats.candidates_found = total_candidates_.load();
    stats.best_match_bits  = best_tracker_.getBestBits();
    stats.hash_rate =
        elapsed.count() > 0 ? static_cast<double>(stats.hashes_computed) / static_cast<double>(elapsed.count()) : 0.0;

    return stats;
}

bool MiningSystem::initializeGPUResources()
{
    std::cout << "[DEBUG] Starting GPU resource initialization\n";

    if (config_.blocks_per_stream <= 0) {
        std::cerr << "Invalid blocks_per_stream: " << config_.blocks_per_stream << "\n";
        return false;
    }

    // Create streams
    streams_.resize(config_.num_streams);
    start_events_.resize(config_.num_streams);
    end_events_.resize(config_.num_streams);

    // Get stream priority range
    int priority_high, priority_low;
    gpuDeviceGetStreamPriorityRange(&priority_low, &priority_high);

    std::cout << "[DEBUG] Creating " << config_.num_streams << " streams\n";

    for (int i = 0; i < config_.num_streams; i++) {
        // Initialize to nullptr first
        streams_[i]    = nullptr;
        int priority   = (i == 0) ? priority_high : priority_low;
        gpuError_t err = gpuStreamCreateWithPriority(&streams_[i], gpuStreamNonBlocking, priority);
        if (err != gpuSuccess) {
            std::cerr << "Failed to create stream " << i << ": " << gpuGetErrorString(err) << "\n";
            // Clean up already created streams
            for (int j = 0; j < i; j++) {
                if (streams_[j]) {
                    gpuStreamDestroy(streams_[j]);
                    streams_[j] = nullptr;
                }
            }
            return false;
        }
        // Verify stream was created
        if (!streams_[i]) {
            std::cerr << "Stream " << i << " is null after creation\n";
            // Clean up
            for (int j = 0; j < i; j++) {
                if (streams_[j]) {
                    gpuStreamDestroy(streams_[j]);
                    streams_[j] = nullptr;
                }
            }
            return false;
        }
        std::cout << "[DEBUG] Created stream " << i << " with handle: " << streams_[i] << "\n";

        // Create events with error checking
        err = gpuEventCreateWithFlags(&start_events_[i], gpuEventDisableTiming);
        if (err != gpuSuccess) {
            std::cerr << "Failed to create start event for stream " << i << ": " << gpuGetErrorString(err) << "\n";
            return false;
        }

        err = gpuEventCreateWithFlags(&end_events_[i], gpuEventDisableTiming);
        if (err != gpuSuccess) {
            std::cerr << "Failed to create end event for stream " << i << ": " << gpuGetErrorString(err) << "\n";
            return false;
        }
    }

    // Validate all streams were created successfully
    if (!validateStreams()) {
        std::cerr << "Stream validation failed after creation\n";
        return false;
    }

    // Initialize memory pools (this will also allocate device jobs)
    if (!initializeMemoryPools()) {
        std::cerr << "Failed to initialize memory pools\n";
        return false;
    }

    std::cout << "Successfully initialized GPU resources for " << config_.num_streams << " streams\n";
    return true;
}

bool MiningSystem::initializeMemoryPools()
{
    std::cout << "[DEBUG] Allocating GPU memory pools\n";

    // Allocate GPU memory pools
    gpu_pools_.resize(config_.num_streams);
    pinned_results_.resize(config_.num_streams);

    // Get memory alignment for platform
    size_t alignment = getMemoryAlignment();

    for (int i = 0; i < config_.num_streams; i++) {
        ResultPool &pool = gpu_pools_[i];
        pool.capacity    = config_.result_buffer_size;

        // Initialize all pointers to nullptr first
        pool.results          = nullptr;
        pool.count            = nullptr;
        pool.nonces_processed = nullptr;
        pool.job_version      = nullptr;

        // Check if stream is valid before using it
        if (!streams_[i]) {
            std::cerr << "Stream " << i << " is invalid before memory allocation\n";
            return false;
        }

        // Allocate device memory with proper alignment
        size_t result_size  = sizeof(MiningResult) * pool.capacity;
        size_t aligned_size = ((result_size + alignment - 1) / alignment) * alignment;

        gpuError_t err = gpuMalloc(&pool.results, aligned_size);
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate GPU results buffer for stream " << i << ": " << gpuGetErrorString(err)
                      << "\n";
            return false;
        }

        // Clear memory immediately
        err = gpuMemsetAsync(pool.results, 0, aligned_size, streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear results buffer: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate count with alignment
        err = gpuMalloc(&pool.count, sizeof(uint32_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate count buffer: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Verify the pointer is valid
        if (!pool.count) {
            std::cerr << "pool.count is null after allocation!\n";
            return false;
        }

        err = gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear count buffer: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate nonces_processed
        err = gpuMalloc(&pool.nonces_processed, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate nonce counter: " << gpuGetErrorString(err) << "\n";
            return false;
        }
        err = gpuMemsetAsync(pool.nonces_processed, 0, sizeof(uint64_t), streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear nonce counter: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate job_version
        err = gpuMalloc(&pool.job_version, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate job version: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        err = gpuMemsetAsync(pool.job_version, 0, sizeof(uint64_t), streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to clear job version: " << gpuGetErrorString(err) << "\n";
            return false;
        }

        // Allocate pinned host memory
        if (config_.use_pinned_memory) {
            err = gpuHostAlloc(&pinned_results_[i], result_size, gpuHostAllocMapped | gpuHostAllocWriteCombined);
            if (err != gpuSuccess) {
                std::cerr << "Warning: Failed to allocate pinned memory, using regular memory\n";
                pinned_results_[i]        = new MiningResult[pool.capacity];
                config_.use_pinned_memory = false;
            }
        } else {
            pinned_results_[i] = new MiningResult[pool.capacity];
        }

        // Synchronize to ensure all allocations are complete
        err = gpuStreamSynchronize(streams_[i]);
        if (err != gpuSuccess) {
            std::cerr << "Failed to synchronize stream " << i << " after allocation: " << gpuGetErrorString(err)
                      << "\n";
            return false;
        }
    }

    // Allocate device memory for jobs
    device_jobs_.resize(config_.num_streams);
    for (int i = 0; i < config_.num_streams; i++) {
        if (!device_jobs_[i].allocate()) {
            std::cerr << "Failed to allocate device job " << i << "\n";
            // Clean up previously allocated jobs
            for (int j = 0; j < i; j++) {
                device_jobs_[j].free();
            }
            return false;
        }
        std::cout << "[DEBUG] Successfully allocated device job " << i << "\n";
    }

    // Initialize job version
    current_job_version_ = 0;

    // Final validation
    for (int i = 0; i < config_.num_streams; i++) {
        if (!gpu_pools_[i].count || !gpu_pools_[i].results || !gpu_pools_[i].nonces_processed ||
            !gpu_pools_[i].job_version) {
            std::cerr << "GPU pool " << i << " has null pointers after allocation\n";
            std::cerr << "  count: " << gpu_pools_[i].count << "\n";
            std::cerr << "  results: " << gpu_pools_[i].results << "\n";
            std::cerr << "  nonces_processed: " << gpu_pools_[i].nonces_processed << "\n";
            std::cerr << "  job_version: " << gpu_pools_[i].job_version << "\n";
            return false;
        }
    }

    std::cout << "Successfully allocated GPU resources for " << config_.num_streams << " streams\n";
    return true;
}

void MiningSystem::cleanup()
{
    std::lock_guard lock(system_mutex_);

    printFinalStats();

    // Synchronize and destroy streams
    for (size_t i = 0; i < streams_.size(); i++) {
        if (streams_[i]) {
            gpuStreamSynchronize(streams_[i]);
            gpuStreamDestroy(streams_[i]);
        }
        if (start_events_[i])
            gpuEventDestroy(start_events_[i]);
        if (end_events_[i])
            gpuEventDestroy(end_events_[i]);
    }

    // Cleanup kernel completion events if they exist
    for (auto &kernel_complete_event : kernel_complete_events_) {
        if (kernel_complete_event) {
            gpuEventDestroy(kernel_complete_event);
        }
    }

    // Free GPU memory
    for (const auto &pool : gpu_pools_) {
        if (pool.results)
            gpuFree(pool.results);
        if (pool.count)
            gpuFree(pool.count);
        if (pool.nonces_processed)
            gpuFree(pool.nonces_processed);
        if (pool.job_version)
            gpuFree(pool.job_version);
    }

    for (auto &job : device_jobs_) {
        job.free();
    }

    // Free pinned memory
    for (const auto &pinned_result : pinned_results_) {
        if (pinned_result) {
            if (config_.use_pinned_memory) {
                gpuFreeHost(pinned_result);
            } else {
                delete[] pinned_result;
            }
        }
    }
}

void MiningSystem::performanceMonitor() const
{
    auto last_update     = std::chrono::steady_clock::now();
    uint64_t last_hashes = 0;

    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now           = std::chrono::steady_clock::now();
        auto elapsed       = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        const uint64_t current_hashes = total_hashes_.load();
        const uint64_t hash_diff      = current_hashes - last_hashes;

        const double instant_rate =
            elapsed.count() > 0 ? static_cast<double>(hash_diff) / static_cast<double>(elapsed.count()) / 1e9 : 0.0;

        const double average_rate = total_elapsed.count() > 0 ? static_cast<double>(current_hashes) /
                                                                    static_cast<double>(total_elapsed.count()) / 1e9
                                                              : 0.0;

        std::cout << "\r[" << total_elapsed.count() << "s] "
                  << "Rate: " << std::fixed << std::setprecision(2) << instant_rate << " GH/s"
                  << " (avg: " << average_rate << " GH/s) | "
                  << "Best: " << best_tracker_.getBestBits() << " bits | "
                  << "Total: " << static_cast<double>(current_hashes) / 1e12 << " TH" << std::flush;

        last_update = now;
        last_hashes = current_hashes;
    }
}

void MiningSystem::performanceMonitorInterruptible(const std::function<bool()> &should_continue) const
{
    auto last_update     = std::chrono::steady_clock::now();
    uint64_t last_hashes = 0;

    while (!g_shutdown && should_continue()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now           = std::chrono::steady_clock::now();
        auto elapsed       = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        const uint64_t current_hashes = total_hashes_.load();
        const uint64_t hash_diff      = current_hashes - last_hashes;

        const double instant_rate =
            elapsed.count() > 0 ? static_cast<double>(hash_diff) / static_cast<double>(elapsed.count()) / 1e9 : 0.0;

        const double average_rate = total_elapsed.count() > 0 ? static_cast<double>(current_hashes) /
                                                                    static_cast<double>(total_elapsed.count()) / 1e9
                                                              : 0.0;

        std::cout << "\r[" << total_elapsed.count() << "s] "
                  << "Rate: " << std::fixed << std::setprecision(2) << instant_rate << " GH/s"
                  << " (avg: " << average_rate << " GH/s) | "
                  << "Best: " << best_tracker_.getBestBits() << " bits | "
                  << "Total: " << static_cast<double>(current_hashes) / 1e12 << " TH" << std::flush;

        last_update = now;
        last_hashes = current_hashes;
    }

    if (!should_continue()) {
        std::cout << "\n[STOPPED] Pool connection lost - mining halted.\n";
    }
}

void MiningSystem::printFinalStats() const
{
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

    const auto hashes = total_hashes_.load();

    std::cout << "\n\nFinal Statistics:\n";
    std::cout << "=====================================\n";
    std::cout << "  Platform: " << getGPUPlatformName() << "\n";
    std::cout << "  GPU: " << device_props_.name << "\n";
    std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " << hashes << " (" << (static_cast<double>(hashes) / 1e9) << " GH)\n";
    if (elapsed.count() > 0) {
        std::cout << "  Average Rate: "
                  << static_cast<double>(total_hashes_.load()) / static_cast<double>(elapsed.count()) / 1e9
                  << " GH/s\n";
    }
    std::cout << "  Best Match: " << best_tracker_.getBestBits() << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates_.load() << "\n";

#ifdef DEBUG_SHA1
    std::lock_guard lock(timing_mutex_);
    timing_stats_.print();
#endif
}

uint64_t MiningSystem::getTotalThreads() const
{
    return static_cast<uint64_t>(config_.num_streams) * static_cast<uint64_t>(config_.blocks_per_stream) *
           static_cast<uint64_t>(config_.threads_per_block);
}

uint64_t MiningSystem::getHashesPerKernel() const
{
    return static_cast<uint64_t>(config_.blocks_per_stream) * static_cast<uint64_t>(config_.threads_per_block) *
           static_cast<uint64_t>(NONCES_PER_THREAD);
}

void MiningSystem::optimizeForGPU()
{
    // Additional GPU-specific optimizations can be added here
    // This is called after vendor detection
}

// Additional methods needed by MiningSystem
uint64_t MiningSystem::runSingleBatch(const MiningJob &job)
{
    // Copy job to all device streams for consistency
    for (int i = 0; i < config_.num_streams; i++) {
        device_jobs_[i].copyFromHost(job);
    }

    // Configure kernel
    KernelConfig kernel_config{};
    kernel_config.blocks             = config_.blocks_per_stream;
    kernel_config.threads_per_block  = config_.threads_per_block;
    kernel_config.stream             = streams_[0];  // Use first stream
    kernel_config.shared_memory_size = 0;

    // Reset nonce counter - IMPORTANT!
    gpuMemsetAsync(gpu_pools_[0].nonces_processed, 0, sizeof(uint64_t), streams_[0]);

    // Launch kernel
    launch_mining_kernel(device_jobs_[0], job.difficulty, job.nonce_offset, gpu_pools_[0], kernel_config,
                         current_job_version_);

    // Wait for completion
    gpuStreamSynchronize(streams_[0]);

    // Get actual nonces processed
    uint64_t actual_nonces = 0;
    gpuMemcpy(&actual_nonces, gpu_pools_[0].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost);

    // Process results
    processResultsOptimized(0);

    // Update total hashes with actual count
    total_hashes_ += actual_nonces;

    return actual_nonces;
}

void MiningSystem::stopMining()
{
    stop_mining_ = true;
    // g_shutdown = true;
}

void MiningSystem::clearResults()
{
    std::lock_guard lock(all_results_mutex_);
    all_results_.clear();
}

void MiningSystem::resetState()
{
    LOG_INFO("RESET", "Resetting mining system state");

    best_tracker_.reset();
    total_hashes_        = 0;
    total_candidates_    = 0;
    current_job_version_ = 0;
    clearResults();
    start_time_ = std::chrono::steady_clock::now();
}

extern "C" void cleanup_mining_system()
{
    if (g_mining_system) {
        g_mining_system.reset();
    }
}

extern "C" MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty)
{
    MiningJob job{};

    // Copy message (32 bytes)
    std::memcpy(job.base_message, message, 32);

    // Convert target hash to uint32_t array (big-endian)
    for (int i = 0; i < 5; i++) {
        job.target_hash[i] =
            (static_cast<uint32_t>(target_hash[i * 4]) << 24) | (static_cast<uint32_t>(target_hash[i * 4 + 1]) << 16) |
            (static_cast<uint32_t>(target_hash[i * 4 + 2]) << 8) | static_cast<uint32_t>(target_hash[i * 4 + 3]);
    }

    job.difficulty   = difficulty;
    job.nonce_offset = 1;

    return job;
}

extern "C" void run_mining_loop(MiningJob job)
{
    if (!g_mining_system) {
        std::cerr << "Mining system not initialized\n";
        return;
    }

    g_mining_system->runMiningLoop(job);
}
