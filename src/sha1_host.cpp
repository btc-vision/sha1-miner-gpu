// sha1_host_fixed.cpp - Production-ready host-side implementation
// Fixes memory management, adds validation, and improves performance

#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>

extern std::atomic<bool> g_shutdown;

// Enhanced mining system with proper resource management
class MiningSystem {
public:
    struct Config {
        int device_id = 0;
        int num_streams = 4;
        int blocks_per_stream = 0; // Auto-calculate
        int threads_per_block = 256;
        bool use_pinned_memory = true;
        size_t result_buffer_size = MAX_CANDIDATES_PER_BATCH;
    };

private:
    Config config_;
    cudaDeviceProp device_props_;

    // CUDA resources
    std::vector<cudaStream_t> streams_;
    std::vector<ResultPool> gpu_pools_;
    std::vector<MiningResult *> pinned_results_;

    // Performance tracking
    std::atomic<uint64_t> total_hashes_{0};
    std::atomic<uint64_t> total_candidates_{0};
    std::chrono::steady_clock::time_point start_time_;

    // Thread management
    std::unique_ptr<std::thread> monitor_thread_;
    std::mutex system_mutex_;

    // Device memory for constant data
    uint32_t *d_message_ = nullptr;
    uint32_t *d_target_ = nullptr;

public:
    MiningSystem(const Config &config = {}) : config_(config), device_props_() {
    }

    ~MiningSystem() {
        cleanup();
    }

    bool initialize() {
        std::lock_guard<std::mutex> lock(system_mutex_);

        // Set and validate device
        cudaError_t err = cudaSetDevice(config_.device_id);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Get device properties
        err = cudaGetDeviceProperties(&device_props_, config_.device_id);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Print device information
        std::cout << "Initializing SHA-1 Near-Collision Miner v2.0\n";
        std::cout << "Device: " << device_props_.name << "\n";
        std::cout << "Compute Capability: " << device_props_.major << "."
                << device_props_.minor << "\n";
        std::cout << "SMs: " << device_props_.multiProcessorCount << "\n";
        std::cout << "Max Threads/Block: " << device_props_.maxThreadsPerBlock << "\n";
        std::cout << "Shared Memory/Block: " << device_props_.sharedMemPerBlock << " bytes\n";
        std::cout << "L2 Cache: " << device_props_.l2CacheSize << " bytes\n\n";

        // Auto-calculate optimal blocks if not specified
        if (config_.blocks_per_stream == 0) {
            // Use 2-4 blocks per SM for better occupancy
            config_.blocks_per_stream = device_props_.multiProcessorCount * 3;
        }

        // Validate thread configuration
        if (config_.threads_per_block % 32 != 0 ||
            config_.threads_per_block > device_props_.maxThreadsPerBlock) {
            std::cerr << "Invalid thread configuration\n";
            return false;
        }

        // Initialize CUDA resources
        if (!initializeCudaResources()) {
            return false;
        }

        // Set up L2 cache persistence for Ampere+
        if (device_props_.major >= 8) {
            size_t l2_cache_size = device_props_.l2CacheSize;
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_cache_size);
        }

        start_time_ = std::chrono::steady_clock::now();

        std::cout << "Mining Configuration:\n";
        std::cout << "  Streams: " << config_.num_streams << "\n";
        std::cout << "  Blocks/Stream: " << config_.blocks_per_stream << "\n";
        std::cout << "  Threads/Block: " << config_.threads_per_block << "\n";
        std::cout << "  Total Threads: " << getTotalThreads() << "\n";
        std::cout << "  Hashes/Kernel: " << getHashesPerKernel() << "\n";
        std::cout << "  Pinned Memory: " << (config_.use_pinned_memory ? "Yes" : "No") << "\n\n";

        return true;
    }

    void runMiningLoop(const MiningJob &job, uint32_t duration_seconds) {
        std::cout << "Starting mining for " << duration_seconds << " seconds...\n";
        std::cout << "Target difficulty: " << job.difficulty << " bits\n\n";

        // Start performance monitor
        monitor_thread_ = std::make_unique<std::thread>(
            &MiningSystem::performanceMonitor, this
        );

        auto end_time = std::chrono::steady_clock::now() +
                        std::chrono::seconds(duration_seconds);

        uint64_t nonce_offset = 0;
        int stream_idx = 0;

        while (std::chrono::steady_clock::now() < end_time && !g_shutdown) {
            // Get current stream and pool
            auto &stream = streams_[stream_idx];
            auto &pool = gpu_pools_[stream_idx];

            // Update job with current nonce offset
            MiningJob current_job = job;
            current_job.nonce_offset = nonce_offset;

            // Configure kernel launch
            KernelConfig config;
            config.blocks = config_.blocks_per_stream;
            config.threads_per_block = config_.threads_per_block;
            config.shared_memory_size = sizeof(uint32_t) * 8 * 7; // For warp reduction
            config.stream = stream;

            // Launch kernel
            launch_mining_kernel(current_job, pool, config);

            // Update counters
            uint64_t hashes_this_launch = getHashesPerKernel();
            nonce_offset += hashes_this_launch;
            total_hashes_ += hashes_this_launch;

            // Process results from previous stream (if ready)
            int prev_stream = (stream_idx - 1 + config_.num_streams) % config_.num_streams;
            processResults(prev_stream);

            // Move to next stream
            stream_idx = (stream_idx + 1) % config_.num_streams;

            // Prevent nonce overflow
            if (nonce_offset < job.nonce_offset) {
                std::cout << "\nNonce space exhausted, resetting...\n";
                nonce_offset = 0;
            }
        }

        // Wait for all streams to complete
        for (auto &stream: streams_) {
            cudaStreamSynchronize(stream);
        }

        // Process final results
        for (int i = 0; i < config_.num_streams; i++) {
            processResults(i);
        }

        // Stop monitor thread
        g_shutdown = true;
        if (monitor_thread_ && monitor_thread_->joinable()) {
            monitor_thread_->join();
        }
    }

    MiningStats getStats() const {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_
        );

        MiningStats stats;
        stats.hashes_computed = total_hashes_.load();
        stats.candidates_found = total_candidates_.load();
        stats.best_match_bits = 0; // Would need to track this
        stats.hash_rate = static_cast<double>(stats.hashes_computed) /
                          static_cast<double>(elapsed.count());

        return stats;
    }

private:
    bool initializeCudaResources() {
        // Create streams with priorities
        streams_.resize(config_.num_streams);
        for (int i = 0; i < config_.num_streams; i++) {
            int priority = (i == 0) ? -1 : 0; // Higher priority for first stream
            cudaError_t err = cudaStreamCreateWithPriority(
                &streams_[i], cudaStreamNonBlocking, priority
            );
            if (err != cudaSuccess) {
                std::cerr << "Failed to create stream " << i << ": "
                        << cudaGetErrorString(err) << "\n";
                return false;
            }
        }

        // Allocate GPU memory pools
        gpu_pools_.resize(config_.num_streams);
        pinned_results_.resize(config_.num_streams);

        for (int i = 0; i < config_.num_streams; i++) {
            ResultPool &pool = gpu_pools_[i];
            pool.capacity = config_.result_buffer_size;

            // Allocate device memory
            cudaError_t err = cudaMalloc(&pool.results,
                                         sizeof(MiningResult) * pool.capacity);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate GPU results buffer\n";
                return false;
            }

            err = cudaMalloc(&pool.count, sizeof(uint32_t));
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate count buffer\n";
                return false;
            }

            // Initialize count to 0
            cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[i]);

            // Allocate pinned host memory for fast transfers
            if (config_.use_pinned_memory) {
                err = cudaHostAlloc(&pinned_results_[i],
                                    sizeof(MiningResult) * pool.capacity,
                                    cudaHostAllocDefault);
                if (err != cudaSuccess) {
                    std::cerr << "Failed to allocate pinned memory\n";
                    // Fall back to regular memory
                    pinned_results_[i] = new MiningResult[pool.capacity];
                }
            } else {
                pinned_results_[i] = new MiningResult[pool.capacity];
            }
        }

        return true;
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(system_mutex_);

        // Signal shutdown
        g_shutdown = true;

        // Wait for monitor thread
        if (monitor_thread_ && monitor_thread_->joinable()) {
            monitor_thread_->join();
        }

        // Synchronize and destroy streams
        for (auto &stream: streams_) {
            if (stream) {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
        }

        // Free GPU memory
        for (auto &pool: gpu_pools_) {
            if (pool.results) cudaFree(pool.results);
            if (pool.count) cudaFree(pool.count);
        }

        // Free pinned memory
        for (auto &results: pinned_results_) {
            if (results) {
                if (config_.use_pinned_memory) {
                    cudaFreeHost(results);
                } else {
                    delete[] results;
                }
            }
        }

        // Free device memory
        if (d_message_) cudaFree(d_message_);
        if (d_target_) cudaFree(d_target_);

        // Print final statistics
        printFinalStats();
    }

    void processResults(int stream_idx) {
        if (cudaStreamQuery(streams_[stream_idx]) != cudaSuccess) {
            return; // Stream not ready
        }

        auto &pool = gpu_pools_[stream_idx];
        auto &results = pinned_results_[stream_idx];

        // Get result count
        uint32_t count;
        cudaMemcpyAsync(&count, pool.count, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, streams_[stream_idx]);
        cudaStreamSynchronize(streams_[stream_idx]);

        if (count == 0) return;

        // Limit to pool capacity
        count = std::min(count, pool.capacity);

        // Copy results
        cudaMemcpyAsync(results, pool.results, sizeof(MiningResult) * count,
                        cudaMemcpyDeviceToHost, streams_[stream_idx]);
        cudaStreamSynchronize(streams_[stream_idx]);

        // Process candidates
        for (uint32_t i = 0; i < count; i++) {
            std::cout << "\n[CANDIDATE] Nonce: 0x" << std::hex << results[i].nonce
                    << " | Bits: " << std::dec << results[i].matching_bits
                    << " | Hash: ";
            for (int j = 0; j < 5; j++) {
                std::cout << std::hex << std::setw(8) << std::setfill('0')
                        << results[i].hash[j];
            }
            std::cout << "\n";
        }

        // Update statistics
        total_candidates_ += count;

        // Reset pool count
        cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), streams_[stream_idx]);
    }

    void performanceMonitor() {
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
                    << "Candidates: " << total_candidates_.load()
                    << " | Total: " << static_cast<double>(current_hashes) / 1e12
                    << " TH" << std::flush;

            last_update = now;
            last_hashes = current_hashes;
        }
    }

    void printFinalStats() {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_
        );

        std::cout << "\n\nFinal Statistics:\n";
        std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
        std::cout << "  Total Hashes: " <<
                static_cast<double>(total_hashes_.load()) / 1e12 << " TH\n";
        std::cout << "  Average Rate: " <<
                static_cast<double>(total_hashes_.load()) /
                static_cast<double>(elapsed.count()) / 1e9 << " GH/s\n";
        std::cout << "  Candidates Found: " << total_candidates_.load() << "\n";
        std::cout << "  Efficiency: " <<
                static_cast<double>(total_candidates_.load()) /
                static_cast<double>(total_hashes_.load()) * 1e9
                << " candidates/GH\n";
    }

    uint64_t getTotalThreads() const {
        return static_cast<uint64_t>(config_.num_streams) *
               static_cast<uint64_t>(config_.blocks_per_stream) *
               static_cast<uint64_t>(config_.threads_per_block);
    }

    uint64_t getHashesPerKernel() const {
        return static_cast<uint64_t>(config_.blocks_per_stream) *
               static_cast<uint64_t>(config_.threads_per_block) *
               static_cast<uint64_t>(NONCES_PER_THREAD);
    }
};

// Global system instance
static std::unique_ptr<MiningSystem> g_mining_system;

// C-style interface functions
extern "C" bool init_mining_system(int device_id) {
    if (g_mining_system) {
        std::cerr << "Mining system already initialized\n";
        return false;
    }

    MiningSystem::Config config;
    config.device_id = device_id;

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
    job.nonce_offset = 0;

    return job;
}

extern "C" void run_mining_loop(MiningJob job, uint32_t duration_seconds) {
    if (!g_mining_system) {
        std::cerr << "Mining system not initialized\n";
        return;
    }

    g_mining_system->runMiningLoop(job, duration_seconds);
}
