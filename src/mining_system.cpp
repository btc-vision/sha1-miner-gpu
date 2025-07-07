#include "mining_system.hpp"
#include "sha1_miner.cuh"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include "utilities.hpp"

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

    // Calculate optimal blocks based on architecture and vendor
    int blocks_per_sm;
    int optimal_threads;

    if (gpu_vendor_ == GPUVendor::AMD) {
        // AMD-specific tuning
        optimal_threads = 256; // AMD typically prefers 256
        // For RDNA (RX 5700 XT), multiProcessorCount may report half the actual CUs
        int actual_cus = device_props_.multiProcessorCount;
        // Check if this is RDNA by wavefront size
        if (device_props_.warpSize == 32) {
            // RDNA has 32-thread wavefronts instead of 64
            // ROCm might report WGPs (Workgroup Processors) instead of CUs
            // Each WGP has 2 CUs in RDNA
            actual_cus *= 2;
            std::cout << "Detected RDNA architecture, adjusting CU count to: " << actual_cus << "\n";
        }
        // AMD GPU architectures
        if (device_props_.major >= 10) {
            // RDNA/RDNA2/RDNA3 (gfx10xx/gfx11xx)
            blocks_per_sm = 32; // Much higher for RDNA
            config_.num_streams = 8;
        } else if (device_props_.major == 9) {
            // GCN5/Vega (gfx900/gfx906)
            blocks_per_sm = 16;
            config_.num_streams = 4;
        } else {
            // Older GCN
            blocks_per_sm = 8;
            config_.num_streams = 2;
        }

        // Use actual CU count for calculations
        config_.blocks_per_stream = actual_cus * blocks_per_sm;
    } else if (gpu_vendor_ == GPUVendor::NVIDIA) {
        // NVIDIA-specific tuning
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
    } else {
        // Unknown vendor - use conservative defaults
        blocks_per_sm = 4;
        optimal_threads = 256;
        config_.num_streams = 4;
    }

    // Adjust based on register and shared memory limits
    int max_threads_per_sm = device_props_.maxThreadsPerMultiProcessor;
    int max_blocks_per_sm = max_threads_per_sm / optimal_threads;
    if (blocks_per_sm > max_blocks_per_sm) {
        blocks_per_sm = max_blocks_per_sm;
    }

    // Set configuration
    config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
    config_.threads_per_block = optimal_threads;

    // For very large GPUs, limit total blocks to avoid scheduling overhead
    int max_total_blocks = (gpu_vendor_ == GPUVendor::AMD) ? 2048 : 2048; // Same for both now
    if (config_.blocks_per_stream > max_total_blocks) {
        config_.blocks_per_stream = max_total_blocks;
    }

    // Adjust number of streams based on SM/CU count for both vendors
    if (device_props_.multiProcessorCount >= 80) {
        // High-end GPUs (RTX 4090, RX 7900 XTX, etc.)
        config_.num_streams = 16;
    } else if (device_props_.multiProcessorCount >= 40) {
        // Mid-high GPUs (RTX 3070, RX 6800, etc.)
        config_.num_streams = 8;
    } else if (device_props_.multiProcessorCount >= 20) {
        // Mid-range GPUs
        config_.num_streams = 4;
    }

    // Adjust streams based on available memory
    size_t free_mem, total_mem;
    gpuMemGetInfo(&free_mem, &total_mem);
    size_t mem_per_stream = sizeof(MiningResult) * config_.result_buffer_size +
                            (config_.blocks_per_stream * config_.threads_per_block * sizeof(uint32_t) * 5);
    int max_streams_by_memory = free_mem / (mem_per_stream * 2); // Use at most 50% of free memory
    if (config_.num_streams > max_streams_by_memory && max_streams_by_memory > 0) {
        config_.num_streams = max_streams_by_memory;
    }

    // Ensure we have at least 1 stream
    if (config_.num_streams < 1) {
        config_.num_streams = 1;
    }

    // Result buffer size
    config_.result_buffer_size = 128;

    // Special optimizations for specific GPUs
    std::string gpu_name = device_props_.name;

    // NVIDIA specific models
    if (gpu_name.find("4090") != std::string::npos || gpu_name.find("4080") != std::string::npos) {
        // RTX 4090/4080 specific
        config_.threads_per_block = 512; // These GPUs love high thread counts
        blocks_per_sm = 16;
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
    } else if (gpu_name.find("A100") != std::string::npos || gpu_name.find("H100") != std::string::npos) {
        // Data center GPUs
        config_.threads_per_block = 512;
        blocks_per_sm = 32; // These can handle extreme occupancy
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
        config_.num_streams = 32;
    }
    // AMD specific models
    else if (gpu_name.find("7900") != std::string::npos) {
        // RX 7900 XTX/XT (RDNA3)
        config_.threads_per_block = 256;
        blocks_per_sm = 16;
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
        config_.num_streams = 16;
    } else if (gpu_name.find("6900") != std::string::npos || gpu_name.find("6800") != std::string::npos) {
        // RX 6900/6800 (RDNA2)
        config_.threads_per_block = 256;
        blocks_per_sm = 12;
        config_.blocks_per_stream = device_props_.multiProcessorCount * blocks_per_sm;
    }

    // Ensure we don't exceed device limits
    if (config_.threads_per_block > device_props_.maxThreadsPerBlock) {
        config_.threads_per_block = device_props_.maxThreadsPerBlock;
    }

    std::cout << "Auto-tuned configuration for " << device_props_.name << ":\n";
    std::cout << "  Compute Capability: " << device_props_.major << "." << device_props_.minor << "\n";
    std::cout << "  SMs/CUs: " << device_props_.multiProcessorCount << "\n";
    std::cout << "  Blocks per SM/CU: " << blocks_per_sm << "\n";
    std::cout << "  Blocks per stream: " << config_.blocks_per_stream << "\n";
    std::cout << "  Threads per block: " << config_.threads_per_block << "\n";
    std::cout << "  Number of streams: " << config_.num_streams << "\n";
    std::cout << "  Total concurrent threads: " <<
            (config_.blocks_per_stream * config_.threads_per_block * config_.num_streams) << "\n\n";
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

void MiningSystem::runMiningLoop(const MiningJob &job, uint32_t duration_seconds) {
    std::cout << "Starting mining for " << duration_seconds << " seconds...\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Target hash: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::hex << std::setw(8) << std::setfill('0')
                << job.target_hash[i] << " ";
    }
    std::cout << "\n" << std::dec;
    std::cout << "Only new best matches will be reported.\n";
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

    // Start performance monitor
    monitor_thread_ = std::make_unique<std::thread>(
        &MiningSystem::performanceMonitor, this
    );

    auto end_time = std::chrono::steady_clock::now() +
                    std::chrono::seconds(duration_seconds);

    // Initialize per-stream data
    struct StreamData {
        uint64_t nonce_offset;
        bool busy;
        std::chrono::high_resolution_clock::time_point launch_time;
        uint64_t last_nonces_processed; // Add tracking per stream
    };
    std::vector<StreamData> stream_data(config_.num_streams);
    // Initialize stream tracking
    for (int i = 0; i < config_.num_streams; i++) {
        stream_data[i].last_nonces_processed = 0;
        // Reset the actual nonces counter for each stream
        gpuMemsetAsync(gpu_pools_[i].nonces_processed, 0, sizeof(uint64_t), streams_[i]);
    }

    // Nonce distribution
    uint64_t nonce_stride = getHashesPerKernel();
    uint64_t global_nonce_offset = 1; // Start from 1

    // Mining loop
    int current_stream = 0;
    uint64_t kernels_launched = 0;

    while (std::chrono::steady_clock::now() < end_time && !g_shutdown) {
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

        if (stream_data[current_stream].busy) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
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

        // Don't update total_hashes_ here - wait for actual count
        kernels_launched++;
        global_nonce_offset += nonce_stride;

        // Move to next stream
        current_stream = (current_stream + 1) % config_.num_streams;
    }

    // Wait for all streams to complete and get final counts
    for (int i = 0; i < config_.num_streams; i++) {
        gpuStreamSynchronize(streams_[i]);
        // Get final actual nonces processed
        uint64_t actual_nonces = 0;
        gpuMemcpy(&actual_nonces, gpu_pools_[i].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost);
        // Update total with any remaining work
        uint64_t nonces_this_kernel = actual_nonces - stream_data[i].last_nonces_processed;
        total_hashes_ += nonces_this_kernel;

        processResultsOptimized(i);
    }

    // Stop monitor thread
    g_shutdown = true;
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }

    std::cout << "\n\nMining completed. Kernels launched: " << kernels_launched << "\n";
    std::cout << "Actual hashes computed: " << static_cast<double>(total_hashes_) / 1e9 << " GH\n";
    std::cout << "Final best match: " << best_tracker_.getBestBits() << " bits\n";
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
    count = std::min(count, pool.capacity);

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

    // Process results - only report new bests
    for (uint32_t i = 0; i < count; i++) {
        if (results[i].nonce == 0) continue;

        if (best_tracker_.isNewBest(results[i].matching_bits)) {
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
        }

        total_candidates_++;
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

        // Allocate device memory with proper alignment
        size_t result_size = sizeof(MiningResult) * pool.capacity;
        size_t aligned_size = ((result_size + alignment - 1) / alignment) * alignment;

        gpuError_t err = gpuMalloc(&pool.results, aligned_size);
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate GPU results buffer\n";
            return false;
        }

        err = gpuMalloc(&pool.count, sizeof(uint32_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate count buffer\n";
            return false;
        }

        gpuMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[i]);

        err = gpuMalloc(&pool.nonces_processed, sizeof(uint64_t));
        if (err != gpuSuccess) {
            std::cerr << "Failed to allocate nonce counter\n";
            return false;
        }
        gpuMemsetAsync(pool.nonces_processed, 0, sizeof(uint64_t), streams_[i]);

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
    }

    // Allocate device memory for jobs
    device_jobs_.resize(config_.num_streams);
    for (int i = 0; i < config_.num_streams; i++) {
        device_jobs_[i].allocate();
    }

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
    return static_cast<uint64_t>(config_.blocks_per_stream) *
           static_cast<uint64_t>(config_.threads_per_block) *
           static_cast<uint64_t>(NONCES_PER_THREAD);

    return config_.blocks_per_stream * config_.threads_per_block * NONCES_PER_THREAD;
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

extern "C" void run_mining_loop(MiningJob job, uint32_t duration_seconds) {
    if (!g_mining_system) {
        std::cerr << "Mining system not initialized\n";
        return;
    }

    g_mining_system->runMiningLoop(job, duration_seconds);
}
