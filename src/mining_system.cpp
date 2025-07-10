#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include "gpu_architecture.hpp"
#include "utilities.hpp"
#include "mining_system.hpp"
#include "sha1_miner.cuh"

// Global system instance
std::unique_ptr<MiningSystem> g_mining_system;

// BestResultTracker implementation
BestResultTracker::BestResultTracker() : best_bits_(0) {
}

bool BestResultTracker::isNewBest(uint32_t matching_bits) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (matching_bits > best_bits_) {
        best_bits_ = matching_bits;
        return true;
    }
    return false;
}

uint32_t BestResultTracker::getBestBits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return best_bits_;
}

void BestResultTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    best_bits_ = 0;
}

// TimingStats implementation
void MiningSystem::TimingStats::reset() {
    kernel_launch_time_ms = 0;
    kernel_execution_time_ms = 0;
    result_copy_time_ms = 0;
    total_kernel_time_ms = 0;
    kernel_count = 0;
}

void MiningSystem::TimingStats::print() const {
    if (kernel_count == 0) return;
    std::cout << "\n[TIMING STATS] After " << kernel_count << " kernels:\n";
    std::cout << "  Average kernel execution: " << (kernel_execution_time_ms / kernel_count) << " ms\n";
    std::cout << "  Average result copy: " << (result_copy_time_ms / kernel_count) << " ms\n";
    std::cout << "  Total accumulated time: " << total_kernel_time_ms << " ms\n";
}

// MiningSystem implementation
MiningSystem::MiningSystem(const Config &config)
    : config_(config), device_props_(), gpu_vendor_(GPUVendor::UNKNOWN), best_tracker_() {
}

MiningSystem::~MiningSystem() {
    cleanup();
}

GPUVendor MiningSystem::detectGPUVendor() const {
    std::string device_name = device_props_.name;
    // Convert to lowercase for comparison
    std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
    // Check for NVIDIA GPUs
    if (device_name.find("nvidia") != std::string::npos ||
        device_name.find("geforce") != std::string::npos ||
        device_name.find("quadro") != std::string::npos ||
        device_name.find("tesla") != std::string::npos ||
        device_name.find("titan") != std::string::npos ||
        device_name.find("rtx") != std::string::npos ||
        device_name.find("gtx") != std::string::npos) {
        return GPUVendor::NVIDIA;
    }
    // Check for AMD GPUs
    if (device_name.find("amd") != std::string::npos ||
        device_name.find("radeon") != std::string::npos ||
        device_name.find("vega") != std::string::npos ||
        device_name.find("polaris") != std::string::npos ||
        device_name.find("navi") != std::string::npos ||
        device_name.find("rdna") != std::string::npos ||
        device_name.find("gfx") != std::string::npos) {
        return GPUVendor::AMD;
    }
    return GPUVendor::UNKNOWN;
}

void MiningSystem::autoTuneParameters() {
    std::cout << "Auto-tuning mining parameters...\n";

    // Detect GPU vendor
    gpu_vendor_ = detectGPUVendor();
    std::cout << "Detected GPU vendor: ";
    switch (gpu_vendor_) {
        case GPUVendor::NVIDIA:
            std::cout << "NVIDIA\n";
            break;
        case GPUVendor::AMD:
            std::cout << "AMD\n";
            break;
        default:
            std::cout << "Unknown (using generic optimization)\n";
            break;
    }

    int blocks_per_sm;
    int optimal_threads;

    if (gpu_vendor_ == GPUVendor::AMD) {
#ifdef USE_HIP
        // Use enhanced AMD detection
        AMDArchitecture arch = AMDGPUDetector::detectArchitecture(device_props_);

        // Store the architecture for later use (add this member to class)
        detected_arch_ = arch;  // You'll need to add AMDArchitecture detected_arch_; to the class

        // Print detailed architecture info
        printAMDArchitectureInfo(device_props_);

        // Check for known issues
        if (AMDGPUDetector::hasKnownIssues(arch, device_props_.name)) {
            std::cout << "\nWARNING: This GPU may have compatibility issues.\n";
            std::cout << "Consider updating ROCm/drivers or using reduced settings.\n\n";
        }

        // Use architecture-specific configuration
        AMDGPUDetector::configureForArchitecture(config_, device_props_, arch);

        // IMPORTANT: After configureForArchitecture, we may need to override for specific GPUs

        // Special handling for RDNA4 - ensure we're not limiting it
        if (arch == AMDArchitecture::RDNA4) {
            // Force more aggressive settings for RDNA4
            config_.threads_per_block = 256;

            // Don't let the generic limiter cap RDNA4
            int rdna4_max_blocks = 4096;  // RDNA4 can handle more
            if (config_.blocks_per_stream > rdna4_max_blocks) {
                config_.blocks_per_stream = rdna4_max_blocks;
            }

            // Ensure adequate streams for RDNA4
            if (config_.num_streams < 16) {
                config_.num_streams = 16;
            }

            // Larger result buffer for RDNA4
            config_.result_buffer_size = 1024;
        }

        // Special handling for specific GPUs
        std::string gpu_name = device_props_.name;
        if (gpu_name.find("7900") != std::string::npos) {
            // RX 7900 XTX/XT specific tuning
            std::cout << "Applying RX 7900 series optimizations\n";
            config_.threads_per_block = 256;
            if (config_.blocks_per_stream > 1536) {
                config_.blocks_per_stream = 1536;
            }
        }
#else
        // Fallback for non-HIP builds
        optimal_threads = 256;
        blocks_per_sm = 8;
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
        config_.threads_per_block = optimal_threads;
        config_.num_streams = 4;
#endif
    } else if (gpu_vendor_ == GPUVendor::NVIDIA) {
        // NVIDIA-specific tuning (existing code)
        if (device_props_.major >= 8) {
            // Ampere and newer (RTX 30xx, 40xx, A100, etc.)
            blocks_per_sm = 16;
            optimal_threads = 256;
            config_.num_streams = 16;
        } else if (device_props_.major == 7) {
            if (device_props_.minor >= 5) {
                // Turing (RTX 20xx, T4)
                blocks_per_sm = 8;
                optimal_threads = 256;
                config_.num_streams = 8;
            } else {
                // Volta (V100, Titan V)
                blocks_per_sm = 8;
                optimal_threads = 256;
                config_.num_streams = 8;
            }
        } else if (device_props_.major == 6) {
            // Pascal (GTX 10xx, P100)
            blocks_per_sm = 8;
            optimal_threads = 256;
            config_.num_streams = 4;
        } else {
            // Maxwell and older
            blocks_per_sm = 4;
            optimal_threads = 128;
            config_.num_streams = 2;
        }

        // Set configuration for NVIDIA
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
        config_.threads_per_block = optimal_threads;
    } else {
        // Unknown vendor - use conservative defaults
        blocks_per_sm = 4;
        optimal_threads = 256;
        config_.num_streams = 4;
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
        config_.threads_per_block = optimal_threads;
    }

    // Common adjustments for all vendors

    // Ensure we don't exceed device limits
    if (config_.threads_per_block > device_props_.maxThreadsPerBlock) {
        config_.threads_per_block = device_props_.maxThreadsPerBlock;
    }

    // Architecture-specific block limits
    int max_total_blocks = 2048;  // Default

#ifdef USE_HIP
    if (gpu_vendor_ == GPUVendor::AMD) {
        // AMD-specific limits based on architecture
        if (detected_arch_ == AMDArchitecture::RDNA4) {
            max_total_blocks = 4096;  // RDNA4 can handle more
        } else if (detected_arch_ == AMDArchitecture::RDNA3) {
            max_total_blocks = 3072;  // RDNA3 limit
        } else if (detected_arch_ == AMDArchitecture::RDNA2) {
            max_total_blocks = 2048;  // RDNA2 limit
        }
    }
#endif

    if (config_.blocks_per_stream > max_total_blocks) {
        std::cout << "Capping blocks from " << config_.blocks_per_stream
                  << " to " << max_total_blocks << " for architecture\n";
        config_.blocks_per_stream = max_total_blocks;
    }

    // Adjust streams based on available memory
    size_t free_mem, total_mem;
    (void)gpuMemGetInfo(&free_mem, &total_mem);

    // Calculate memory per stream more accurately
    size_t result_buffer_mem = sizeof(MiningResult) * config_.result_buffer_size;
    size_t working_mem = config_.blocks_per_stream * config_.threads_per_block * 256; // Rough estimate
    size_t mem_per_stream = result_buffer_mem + working_mem + (1024 * 1024); // Add 1MB buffer

    int max_streams_by_memory = (free_mem * 0.8) / mem_per_stream; // Use 80% of free memory

    if (config_.num_streams > max_streams_by_memory && max_streams_by_memory > 0) {
        std::cout << "Reducing streams from " << config_.num_streams
                  << " to " << max_streams_by_memory << " due to memory constraints\n";
        config_.num_streams = max_streams_by_memory;
    }

    // Ensure we have at least 1 stream
    if (config_.num_streams < 1) {
        config_.num_streams = 1;
    }

    // Architecture-specific result buffer sizing
#ifdef USE_HIP
    if (gpu_vendor_ == GPUVendor::AMD && detected_arch_ == AMDArchitecture::RDNA4) {
        // RDNA4 gets larger buffers
        if (config_.result_buffer_size < 512) {
            config_.result_buffer_size = 512;
        }
    } else {
        // Default buffer size
        if (config_.result_buffer_size == 0) {
            config_.result_buffer_size = 128;
        }
    }
#else
    // Default for NVIDIA
    if (config_.result_buffer_size == 0) {
        config_.result_buffer_size = 128;
    }
#endif

    // Print final configuration
    std::cout << "\nAuto-tuned configuration for " << device_props_.name << ":\n";
    std::cout << "  Compute Capability: " << device_props_.major << "." << device_props_.minor << "\n";
    std::cout << "  SMs/CUs: " << device_props_.multiProcessorCount << "\n";

    if (gpu_vendor_ == GPUVendor::NVIDIA) {
        std::cout << "  Blocks per SM: " << (config_.blocks_per_stream / device_props_.multiProcessorCount) << "\n";
    } else if (gpu_vendor_ == GPUVendor::AMD) {
        std::cout << "  Blocks per CU: " << (config_.blocks_per_stream / device_props_.multiProcessorCount) << "\n";
#ifdef USE_HIP
        std::cout << "  Architecture: " << AMDGPUDetector::getArchitectureName(detected_arch_) << "\n";
#endif
    }

    std::cout << "  Blocks per stream: " << config_.blocks_per_stream << "\n";
    std::cout << "  Threads per block: " << config_.threads_per_block << "\n";
    std::cout << "  Number of streams: " << config_.num_streams << "\n";
    std::cout << "  Result buffer size: " << config_.result_buffer_size << "\n";
    std::cout << "  Total concurrent threads: " <<
            (config_.blocks_per_stream * config_.threads_per_block * config_.num_streams) << "\n";

    // Calculate expected memory usage
    size_t total_mem_usage = config_.num_streams * mem_per_stream;
    std::cout << "  Estimated memory usage: " << (total_mem_usage / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Available memory: " << (free_mem / (1024.0 * 1024.0)) << " MB\n\n";
}

bool MiningSystem::initialize() {
    std::lock_guard<std::mutex> lock(system_mutex_);

    // Set device
    gpuError_t err = gpuSetDevice(config_.device_id);
    if (err != gpuSuccess) {
        std::cerr << "Failed to set GPU device: " << gpuGetErrorString(err) << "\n";
        return false;
    }

    // Get device properties
    err = gpuGetDeviceProperties(&device_props_, config_.device_id);
    if (err != gpuSuccess) {
        std::cerr << "Failed to get device properties: " << gpuGetErrorString(err) << "\n";
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
    std::cout << "Total Global Memory: " << (device_props_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n\n";

    // Auto-tune parameters if blocks not specified
    if (config_.blocks_per_stream == 0) {
        autoTuneParameters();
    }

    // Validate thread configuration
    if (config_.threads_per_block % device_props_.warpSize != 0 ||
        config_.threads_per_block > device_props_.maxThreadsPerBlock) {
        std::cerr << "Invalid thread configuration\n";
        return false;
    }

    // Initialize GPU resources
    if (!initializeGPUResources()) {
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
    std::cout << "  Hashes/Kernel: " << getHashesPerKernel() << " (~"
            << (getHashesPerKernel() / 1e9) << " GH)\n\n";

    return true;
}

void MiningSystem::runMiningLoopInterruptible(const MiningJob &job, std::function<bool()> should_continue) {
    std::cout << "Starting interruptible mining...\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Target hash: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::hex << std::setw(8) << std::setfill('0')
                << job.target_hash[i] << " ";
    }
    std::cout << "\n" << std::dec;
    std::cout << "Mining will stop when connection is lost.\n";
    std::cout << "=====================================\n\n";

    // Copy job to device
    for (int i = 0; i < config_.num_streams; i++) {
        device_jobs_[i].copyFromHost(job);
    }

    // Reset flags and counters
    stop_mining_ = false;
    g_shutdown = false;
    best_tracker_.reset();
    total_hashes_ = 0;
    clearResults();

    // Start performance monitor with connection check
    monitor_thread_ = std::make_unique<std::thread>([this, should_continue]() {
        if (!should_continue()) {
            std::cout << "\n[STOPPED] Pool connection lost - mining halted.\n";
        }
    });

    // Initialize per-stream data
    struct StreamData {
        uint64_t nonce_offset;
        bool busy;
        std::chrono::high_resolution_clock::time_point launch_time;
        uint64_t last_nonces_processed;
    };
    std::vector<StreamData> stream_data(config_.num_streams);

    // Initialize stream tracking
    for (int i = 0; i < config_.num_streams; i++) {
        stream_data[i].last_nonces_processed = 0;
        gpuMemsetAsync(gpu_pools_[i].nonces_processed, 0, sizeof(uint64_t), streams_[i]);
    }

    // Nonce distribution
    uint64_t nonce_stride = getHashesPerKernel();
    uint64_t global_nonce_offset = 1; // Start from 1

    // Mining loop - runs until shutdown OR connection lost
    int current_stream = 0;
    uint64_t kernels_launched = 0;

    while (!g_shutdown && !stop_mining_ && should_continue()) {
        // Find next available stream
        int attempts = 0;
        while (stream_data[current_stream].busy && attempts < config_.num_streams) {
            gpuError_t status = gpuStreamQuery(streams_[current_stream]);
            if (status == gpuSuccess) {
                // Stream completed - get actual nonces processed
                uint64_t actual_nonces = 0;
                gpuMemcpyAsync(&actual_nonces, gpu_pools_[current_stream].nonces_processed, sizeof(uint64_t),
                               gpuMemcpyDeviceToHost, streams_[current_stream]);
                gpuStreamSynchronize(streams_[current_stream]);

                // Update total with actual work done
                uint64_t nonces_this_kernel = actual_nonces - stream_data[current_stream].last_nonces_processed;
                total_hashes_ += nonces_this_kernel;
                stream_data[current_stream].last_nonces_processed = actual_nonces;

                // Process results
                processResultsOptimized(current_stream);
                stream_data[current_stream].busy = false;

                // Update timing stats
                auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::high_resolution_clock::now() -
                                       stream_data[current_stream].launch_time
                                   ).count() / 1000.0;

                std::lock_guard<std::mutex> lock(timing_mutex_);
                timing_stats_.kernel_execution_time_ms += kernel_time;
                timing_stats_.kernel_count++;
            }

            current_stream = (current_stream + 1) % config_.num_streams;
            attempts++;
        }

        // Check if we should stop before launching new work
        if (!should_continue()) {
            std::cout << "\n[MINING] Connection lost, stopping work generation...\n";
            break;
        }

        if (stream_data[current_stream].busy) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // Configure kernel
        KernelConfig config;
        config.blocks = config_.blocks_per_stream;
        config.threads_per_block = config_.threads_per_block;
        config.stream = streams_[current_stream];
        config.shared_memory_size = 0;

        // Launch kernel
        auto launch_start = std::chrono::high_resolution_clock::now();

        launch_mining_kernel(
            device_jobs_[current_stream],
            job.difficulty,
            global_nonce_offset,
            gpu_pools_[current_stream],
            config
        );

        stream_data[current_stream].launch_time = launch_start;
        stream_data[current_stream].busy = true;
        stream_data[current_stream].nonce_offset = global_nonce_offset;

        kernels_launched++;
        global_nonce_offset += nonce_stride;

        // Move to next stream
        current_stream = (current_stream + 1) % config_.num_streams;
    }

    // Stop flag was set - wait for all streams to complete
    std::cout << "\n[MINING] Waiting for active streams to complete...\n";
    for (int i = 0; i < config_.num_streams; i++) {
        gpuStreamSynchronize(streams_[i]);

        // Get final nonce counts
        uint64_t actual_nonces = 0;
        gpuMemcpy(&actual_nonces, gpu_pools_[i].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost);
        uint64_t nonces_this_kernel = actual_nonces - stream_data[i].last_nonces_processed;
        total_hashes_ += nonces_this_kernel;

        // Process any remaining results
        processResultsOptimized(i);
    }

    // Stop monitor thread
    g_shutdown = true;
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

    std::cout << "\n\nMining stopped. Kernels launched: " << kernels_launched << "\n";
    std::cout << "Actual hashes computed: " << static_cast<double>(total_hashes_) / 1e9 << " GH\n";
    std::cout << "Final best match: " << best_tracker_.getBestBits() << " bits\n";

    if (!should_continue()) {
        std::cout << "Mining stopped due to lost pool connection.\n";
    }
}

void MiningSystem::runMiningLoop(const MiningJob &job) {
    std::cout << "Starting infinite mining...\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Target hash: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::hex << std::setw(8) << std::setfill('0')
                << job.target_hash[i] << " ";
    }
    std::cout << "\n" << std::dec;
    std::cout << "Only new best matches will be reported.\n";
    std::cout << "Press Ctrl+C to stop mining.\n";
    std::cout << "=====================================\n\n";

    // Copy job to device
    for (int i = 0; i < config_.num_streams; i++) {
        device_jobs_[i].copyFromHost(job);
    }

    // Reset shutdown flag and best tracker
    g_shutdown = false;
    best_tracker_.reset();
    // Reset actual hash counter
    total_hashes_ = 0;
    // Clear accumulated results
    clearResults();

    // Start performance monitor
    monitor_thread_ = std::make_unique<std::thread>(
        &MiningSystem::performanceMonitor, this
    );

    // Initialize per-stream data
    std::vector<StreamData> stream_data(config_.num_streams);

    // Create events for each stream
    kernel_complete_events_.resize(config_.num_streams);
    kernel_launch_times_.resize(config_.num_streams);

    for (int i = 0; i < config_.num_streams; i++) {
        gpuEventCreateWithFlags(&kernel_complete_events_[i], gpuEventDisableTiming);
        stream_data[i].last_nonces_processed = 0;
        stream_data[i].busy = false;
        gpuMemsetAsync(gpu_pools_[i].nonces_processed, 0, sizeof(uint64_t), streams_[i]);
    }

    // Nonce distribution
    uint64_t nonce_stride = getHashesPerKernel();
    uint64_t global_nonce_offset = 1; // Start from 1

    // Mining loop - runs until g_shutdown is set
    uint64_t kernels_launched = 0;

    // Launch initial kernels on all streams to maximize GPU utilization
    for (int i = 0; i < config_.num_streams; i++) {
        launchKernelOnStream(i, global_nonce_offset, job);
        stream_data[i].nonce_offset = global_nonce_offset;
        stream_data[i].busy = true;
        global_nonce_offset += nonce_stride;
        kernels_launched++;
    }

    // Main mining loop
    while (!g_shutdown) {
        // Check for completed kernels using events
        bool found_completed = false;
        int completed_stream = -1;

        // First, do a quick non-blocking check of all events
        for (int i = 0; i < config_.num_streams; i++) {
            if (!stream_data[i].busy) continue;

            gpuError_t status = gpuEventQuery(kernel_complete_events_[i]);
            if (status == gpuSuccess) {
                completed_stream = i;
                found_completed = true;
                break;
            } else if (status != gpuErrorNotReady) {
                std::cerr << "Event query error: " << gpuGetErrorString(status) << "\n";
            }
        }

        if (!found_completed) {
            // No kernels completed yet, sleep briefly to avoid burning CPU
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        // Process the completed stream
        processStreamResults(completed_stream, stream_data[completed_stream]);
        stream_data[completed_stream].busy = false;

        // Launch new work on this stream
        if (!g_shutdown) {
            launchKernelOnStream(completed_stream, global_nonce_offset, job);
            stream_data[completed_stream].nonce_offset = global_nonce_offset;
            stream_data[completed_stream].busy = true;
            global_nonce_offset += nonce_stride;
            kernels_launched++;
        }
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

    // Stop monitor thread
    g_shutdown = true;
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

    std::cout << "\n\nMining stopped. Kernels launched: " << kernels_launched << "\n";
    std::cout << "Actual hashes computed: " << static_cast<double>(total_hashes_) / 1e9 << " GH\n";
    std::cout << "Final best match: " << best_tracker_.getBestBits() << " bits\n";
}

void MiningSystem::launchKernelOnStream(int stream_idx, uint64_t nonce_offset, const MiningJob &job) {
    // Configure kernel
    KernelConfig config;
    config.blocks = config_.blocks_per_stream;
    config.threads_per_block = config_.threads_per_block;
    config.stream = streams_[stream_idx];
    config.shared_memory_size = 0;

    // Record launch time for performance tracking
    kernel_launch_times_[stream_idx] = std::chrono::high_resolution_clock::now();

    // Launch the mining kernel
    launch_mining_kernel(
        device_jobs_[stream_idx],
        job.difficulty,
        nonce_offset,
        gpu_pools_[stream_idx],
        config
    );

    // Record event when kernel completes
    gpuEventRecord(kernel_complete_events_[stream_idx], streams_[stream_idx]);
}

void MiningSystem::processStreamResults(int stream_idx, StreamData &stream_data) {
    // Get actual nonces processed
    uint64_t actual_nonces = 0;
    gpuMemcpyAsync(&actual_nonces, gpu_pools_[stream_idx].nonces_processed,
                   sizeof(uint64_t), gpuMemcpyDeviceToHost, streams_[stream_idx]);

    // Ensure the copy is complete
    gpuStreamSynchronize(streams_[stream_idx]);

    // Update hash count
    uint64_t nonces_this_kernel = actual_nonces - stream_data.last_nonces_processed;
    total_hashes_ += nonces_this_kernel;
    stream_data.last_nonces_processed = actual_nonces;

    // Process mining results
    processResultsOptimized(stream_idx);

    // Update timing statistics
    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - kernel_launch_times_[stream_idx]
    ).count() / 1000.0;

    {
        std::lock_guard<std::mutex> lock(timing_mutex_);
        timing_stats_.kernel_execution_time_ms += kernel_time;
        timing_stats_.kernel_count++;
    }
}

void MiningSystem::updateJobLive(const MiningJob &job, uint64_t job_version) {
    LOG_INFO("MINING", "Updating job to version ", job_version);

    // First, prepare the job update on all device jobs
    for (int i = 0; i < config_.num_streams; i++) {
        // Update the device job with new data
        device_jobs_[i].updateJob(job, job_version);
    }

    // Now synchronize all streams to ensure they see the update
    for (int i = 0; i < config_.num_streams; i++) {
        gpuStreamSynchronize(streams_[i]);
    }

    // Clear the job_updated flag after all streams have synchronized
    // This ensures all blocks have had a chance to see the update
    for (int i = 0; i < config_.num_streams; i++) {
        if (device_jobs_[i].job_update) {
            JobUpdateRequest clear_update;
            clear_update.job_updated = false;
            clear_update.job_version = job_version;
            // Don't need to copy job data again, just clear the flag
            gpuMemcpyAsync(&device_jobs_[i].job_update->job_updated,
                          &clear_update.job_updated,
                          sizeof(bool),
                          gpuMemcpyHostToDevice,
                          streams_[i]);
        }
    }

    // Store current job version
    current_job_version_ = job_version;

    LOG_INFO("MINING", "Live job update to version ", job_version, " completed");
}

void MiningSystem::processResultsOptimized(int stream_idx) {
    auto &pool = gpu_pools_[stream_idx];
    auto &results = pinned_results_[stream_idx];

    // Get result count
    uint32_t count;
    gpuMemcpyAsync(&count, pool.count, sizeof(uint32_t),
                   gpuMemcpyDeviceToHost, streams_[stream_idx]);
    gpuStreamSynchronize(streams_[stream_idx]);

    if (count == 0) return;

    // Limit to capacity
    if (count > pool.capacity) {
        count = pool.capacity;
    }

    // Copy results
    auto copy_start = std::chrono::high_resolution_clock::now();

    gpuMemcpyAsync(results, pool.results, sizeof(MiningResult) * count,
                   gpuMemcpyDeviceToHost, streams_[stream_idx]);
    gpuStreamSynchronize(streams_[stream_idx]);

    auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now() - copy_start
                     ).count() / 1000.0;

    std::lock_guard<std::mutex> lock(timing_mutex_);
    timing_stats_.result_copy_time_ms += copy_time;

    // Process results
    std::vector<MiningResult> valid_results;

    for (uint32_t i = 0; i < count; i++) {
        if (results[i].nonce == 0) continue;

        // Store all valid results
        valid_results.push_back(results[i]);

        // Track best result
        /*if (best_tracker_.isNewBest(results[i].matching_bits)) {
            // Calculate elapsed time
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_
            );

            std::cout << "\n[NEW BEST!] Time: " << elapsed.count() << "s\n";
            std::cout << "  Platform: " << getGPUPlatformName() << "\n";
            std::cout << "  Nonce: 0x" << std::hex << results[i].nonce << std::dec << "\n";
            std::cout << "  Matching bits: " << results[i].matching_bits << "\n";
            std::cout << "  Hash: ";
            for (int j = 0; j < 5; j++) {
                std::cout << std::hex << std::setw(8) << std::setfill('0')
                        << results[i].hash[j];
                if (j < 4) std::cout << " ";
            }
            std::cout << std::dec << "\n";
            std::cout << "  Rate: " << std::fixed << std::setprecision(2)
                    << static_cast<double>(total_hashes_.load()) / elapsed.count() / 1e9
                    << " GH/s\n";
        }*/

        total_candidates_++;
    }

    // Store results for batch processing
    if (!valid_results.empty()) {
        {
            std::lock_guard<std::mutex> results_lock(all_results_mutex_);
            all_results_.insert(all_results_.end(), valid_results.begin(), valid_results.end());
        }

        // Call the callback if set
        {
            std::lock_guard<std::mutex> callback_lock(callback_mutex_);
            if (result_callback_) {
                result_callback_(valid_results);
            }
        }
    }

    // Reset pool count
    gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[stream_idx]);
}

MiningStats MiningSystem::getStats() const {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );

    MiningStats stats;
    stats.hashes_computed = total_hashes_.load();
    stats.candidates_found = total_candidates_.load();
    stats.best_match_bits = best_tracker_.getBestBits();
    stats.hash_rate = static_cast<double>(stats.hashes_computed) /
                      static_cast<double>(elapsed.count());

    return stats;
}

bool MiningSystem::initializeGPUResources() {
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

    for (int i = 0; i < config_.num_streams; i++) {
        int priority = (i == 0) ? priority_high : priority_low;
        gpuError_t err = gpuStreamCreateWithPriority(
            &streams_[i], gpuStreamNonBlocking, priority
        );
        if (err != gpuSuccess) {
            std::cerr << "Failed to create stream " << i << ": "
                    << gpuGetErrorString(err) << "\n";
            return false;
        }

        gpuEventCreateWithFlags(&start_events_[i], gpuEventDisableTiming);
        gpuEventCreateWithFlags(&end_events_[i], gpuEventDisableTiming);
    }

    // Allocate GPU memory pools
    gpu_pools_.resize(config_.num_streams);
    pinned_results_.resize(config_.num_streams);

    // Get memory alignment for platform
    size_t alignment = getMemoryAlignment();

    for (int i = 0; i < config_.num_streams; i++) {
        ResultPool &pool = gpu_pools_[i];
        pool.capacity = config_.result_buffer_size;

        // Initialize all pointers to nullptr first
        pool.results = nullptr;
        pool.count = nullptr;
        pool.nonces_processed = nullptr;
        pool.job_version = nullptr;

        // Allocate device memory with proper alignment
        size_t result_size = sizeof(MiningResult) * pool.capacity;
        size_t aligned_size = ((result_size + alignment - 1) / alignment) * alignment;

        gpuError_t err = gpuMalloc(&pool.results, aligned_size);
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate GPU results buffer: " << gpuGetErrorString(err) << "\n";
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

        gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[i]);

        err = gpuMalloc(&pool.nonces_processed, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate nonce counter: " << gpuGetErrorString(err) << "\n";
            return false;
        }
        gpuMemsetAsync(pool.nonces_processed, 0, sizeof(uint64_t), streams_[i]);

        // ALLOCATE JOB VERSION for live updates
        err = gpuMalloc(&pool.job_version, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate job version: " << gpuGetErrorString(err) << "\n";
            return false;
        }
        gpuMemsetAsync(pool.job_version, 0, sizeof(uint64_t), streams_[i]);

        // Allocate pinned host memory
        if (config_.use_pinned_memory) {
            err = gpuHostAlloc(&pinned_results_[i], result_size,
                               gpuHostAllocMapped | gpuHostAllocWriteCombined);
            if (err != gpuSuccess) {
                std::cerr << "Warning: Failed to allocate pinned memory, using regular memory\n";
                pinned_results_[i] = new MiningResult[pool.capacity];
                config_.use_pinned_memory = false;
            }
        } else {
            pinned_results_[i] = new MiningResult[pool.capacity];
        }

        // Synchronize to ensure all allocations are complete
        gpuStreamSynchronize(streams_[i]);
    }

    // Allocate device memory for jobs WITH job update support
    device_jobs_.resize(config_.num_streams);
    for (int i = 0; i < config_.num_streams; i++) {
        device_jobs_[i].allocate();  // This now includes job_update allocation
    }

    std::cout << "Successfully allocated GPU resources for " << config_.num_streams << " streams\n";
    return true;
}

void MiningSystem::cleanup() {
    std::lock_guard<std::mutex> lock(system_mutex_);

    g_shutdown = true;

    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

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

    // Free GPU memory
    for (auto &pool: gpu_pools_) {
        if (pool.results)
            gpuFree(pool.results);
        if (pool.count)
            gpuFree(pool.count);
        if (pool.nonces_processed)
            gpuFree(pool.nonces_processed);
        if (pool.job_version)
            gpuFree(pool.job_version);
    }

    for (auto &job: device_jobs_) {
        job.free();
    }

    // Free pinned memory
    for (size_t i = 0; i < pinned_results_.size(); i++) {
        if (pinned_results_[i]) {
            if (config_.use_pinned_memory) {
                gpuFreeHost(pinned_results_[i]);
            } else {
                delete[] pinned_results_[i];
            }
        }
    }

    printFinalStats();
}

void MiningSystem::performanceMonitor() {
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_hashes = 0;

    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_update
        );
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - start_time_
        );

        uint64_t current_hashes = total_hashes_.load();
        uint64_t hash_diff = current_hashes - last_hashes;

        double instant_rate = static_cast<double>(hash_diff) /
                              static_cast<double>(elapsed.count()) / 1e9;
        double average_rate = static_cast<double>(current_hashes) /
                              static_cast<double>(total_elapsed.count()) / 1e9;

        std::cout << "\r[" << total_elapsed.count() << "s] "
                << "Rate: " << std::fixed << std::setprecision(2)
                << instant_rate << " GH/s"
                << " (avg: " << average_rate << " GH/s) | "
                << "Best: " << best_tracker_.getBestBits() << " bits | "
                << "Total: " << static_cast<double>(current_hashes) / 1e12
                << " TH" << std::flush;

        last_update = now;
        last_hashes = current_hashes;
    }
}

void MiningSystem::printFinalStats() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );

    std::cout << "\n\nFinal Statistics:\n";
    std::cout << "=====================================\n";
    std::cout << "  Platform: " << getGPUPlatformName() << "\n";
    std::cout << "  GPU: " << device_props_.name << "\n";
    std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " <<
            static_cast<double>(total_hashes_.load()) / 1e12 << " TH\n";
    std::cout << "  Average Rate: " <<
            static_cast<double>(total_hashes_.load()) /
            static_cast<double>(elapsed.count()) / 1e9 << " GH/s\n";
    std::cout << "  Best Match: " << best_tracker_.getBestBits() << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates_.load() << "\n";

#ifdef DEBUG_SHA1
    std::lock_guard<std::mutex> lock(timing_mutex_);
    timing_stats_.print();
#endif
}

uint64_t MiningSystem::getTotalThreads() const {
    return static_cast<uint64_t>(config_.num_streams) *
           static_cast<uint64_t>(config_.blocks_per_stream) *
           static_cast<uint64_t>(config_.threads_per_block);
}

uint64_t MiningSystem::getHashesPerKernel() const {
    // Default calculation
    return static_cast<uint64_t>(config_.blocks_per_stream) *
           static_cast<uint64_t>(config_.threads_per_block) *
           static_cast<uint64_t>(NONCES_PER_THREAD);
}

void MiningSystem::optimizeForGPU() {
    // Additional GPU-specific optimizations can be added here
    // This is called after vendor detection
}

// C-style interface functions
extern "C" bool init_mining_system(int device_id) {
    if (g_mining_system) {
        std::cerr << "Mining system already initialized\n";
        return false;
    }

    MiningSystem::Config config;
    config.device_id = device_id;
    config.num_streams = 4;
    config.threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    config.use_pinned_memory = true;
    config.result_buffer_size = 256;

    g_mining_system = std::make_unique<MiningSystem>(config);
    return g_mining_system->initialize();
}

extern "C" void cleanup_mining_system() {
    if (g_mining_system) {
        g_mining_system.reset();
    }
}

extern "C" MiningJob create_mining_job(
    const uint8_t *message,
    const uint8_t *target_hash,
    uint32_t difficulty
) {
    MiningJob job{};

    // Copy message (32 bytes)
    std::memcpy(job.base_message, message, 32);

    // Convert target hash to uint32_t array (big-endian)
    for (int i = 0; i < 5; i++) {
        job.target_hash[i] = (static_cast<uint32_t>(target_hash[i * 4]) << 24) |
                             (static_cast<uint32_t>(target_hash[i * 4 + 1]) << 16) |
                             (static_cast<uint32_t>(target_hash[i * 4 + 2]) << 8) |
                             static_cast<uint32_t>(target_hash[i * 4 + 3]);
    }

    job.difficulty = difficulty;
    job.nonce_offset = 1;

    return job;
}

extern "C" void run_mining_loop(MiningJob job) {
    if (!g_mining_system) {
        std::cerr << "Mining system not initialized\n";
        return;
    }

    g_mining_system->runMiningLoop(job);
}
