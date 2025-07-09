#ifndef MINING_SYSTEM_HPP
#define MINING_SYSTEM_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <functional>

#include "gpu_platform.hpp"
#include "sha1_miner.cuh"

// Forward declare the global shutdown flag
extern std::atomic<bool> g_shutdown;

/**
 * Callback type for processing mining results in real-time
 */
using MiningResultCallback = std::function<void(const std::vector<MiningResult> &)>;

/**
 * Thread-safe tracker for best mining results
 */
class BestResultTracker {
public:
    BestResultTracker();

    /**
     * Check if this is a new best result and update if so
     * @param matching_bits Number of matching bits in the result
     * @return true if this is a new best, false otherwise
     */
    bool isNewBest(uint32_t matching_bits);

    /**
     * Get the current best number of matching bits
     * @return Current best matching bits count
     */
    uint32_t getBestBits() const;

    /**
     * Reset the tracker to initial state
     */
    void reset();

private:
    mutable std::mutex mutex_;
    uint32_t best_bits_;
};

/**
 * GPU vendor enumeration
 */
enum class GPUVendor {
    NVIDIA,
    AMD,
    UNKNOWN
};

/**
 * Enhanced mining system with proper resource management
 * Supports both NVIDIA and AMD GPUs
 */
class MiningSystem {
public:
    struct Config {
        int device_id;
        int num_streams;
        int blocks_per_stream;
        int threads_per_block;
        bool use_pinned_memory;
        size_t result_buffer_size;
        bool force_generic_kernel;

        // Constructor with default values
        Config()
            : device_id(0)
              , num_streams(4)
              , blocks_per_stream(0) // Auto-calculate
              , threads_per_block(DEFAULT_THREADS_PER_BLOCK)
              , use_pinned_memory(true)
              , result_buffer_size(MAX_CANDIDATES_PER_BATCH)
              , force_generic_kernel(false) {
        }
    };

    void stopMining();

    bool shouldContinueMining() const { return !stop_mining_.load(); }

    /**
     * Set a callback to be called whenever new results are found
     * @param callback Function to call with new results
     */
    void setResultCallback(MiningResultCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        result_callback_ = callback;
    }

    /**
     * Run a single batch of mining without the monitoring thread
     * Used by MultiGPUManager for proper hash tracking
     * @return Number of hashes computed in this batch
     */
    uint64_t runSingleBatch(const MiningJob &job) {
        if (!device_jobs_.empty()) {
            device_jobs_[0].copyFromHost(job);
        }

        // Reset stop flag for this batch
        stop_mining_ = false;

        // Configure kernel
        KernelConfig kernel_config;
        kernel_config.blocks = config_.blocks_per_stream;
        kernel_config.threads_per_block = config_.threads_per_block;
        kernel_config.stream = streams_[0];
        kernel_config.shared_memory_size = 0;

        // Reset nonce counter
        gpuMemsetAsync(gpu_pools_[0].nonces_processed, 0, sizeof(uint64_t), streams_[0]);

        // Launch kernel
        launch_mining_kernel(
            device_jobs_[0],
            job.difficulty,
            job.nonce_offset,
            gpu_pools_[0],
            kernel_config
        );

        // Wait for completion
        gpuStreamSynchronize(streams_[0]);

        // Get actual nonces processed
        uint64_t actual_nonces = 0;
        gpuMemcpy(&actual_nonces, gpu_pools_[0].nonces_processed, sizeof(uint64_t), gpuMemcpyDeviceToHost);

        // Process results
        processResultsOptimized(0);

        // Update total hashes
        total_hashes_ += actual_nonces;

        return actual_nonces;
    }

    /**
        * Get results from the last batch
        * @return Vector of mining results
        */
    std::vector<MiningResult> getLastResults() {
        std::vector<MiningResult> results;
        // Get result count from first pool
        uint32_t count;
        gpuMemcpy(&count, gpu_pools_[0].count, sizeof(uint32_t), gpuMemcpyDeviceToHost);
        if (count > 0 && count <= gpu_pools_[0].capacity) {
            results.resize(count);
            gpuMemcpy(results.data(), gpu_pools_[0].results, sizeof(MiningResult) * count, gpuMemcpyDeviceToHost);
        }
        return results;
    }

    /**
        * Get current configuration
        */
    const Config &getConfig() const {
        return config_;
    }

    /**
     * Reset internal state for new mining session
     */
    void resetState() {
        best_tracker_.reset();
        total_hashes_ = 0;
        total_candidates_ = 0;
        start_time_ = std::chrono::steady_clock::now();
    }

    /**
     * Get all results found since last clear
     * Used for batch processing
     */
    std::vector<MiningResult> getAllResults() {
        std::lock_guard<std::mutex> lock(all_results_mutex_);
        auto results = all_results_;
        all_results_.clear();
        return results;
    }

    /**
     * Clear all accumulated results
     */
    void clearResults() {
        std::lock_guard<std::mutex> lock(all_results_mutex_);
        all_results_.clear();
    }

    // Timing statistics structure
    struct TimingStats {
        double kernel_launch_time_ms = 0;
        double kernel_execution_time_ms = 0;
        double result_copy_time_ms = 0;
        double total_kernel_time_ms = 0;
        int kernel_count = 0;

        void reset();

        void print() const;
    };

    // Constructor with default config
    explicit MiningSystem(const Config &config = Config());

    ~MiningSystem();

    bool initialize();

    void runMiningLoop(const MiningJob &job, uint32_t duration_seconds);

    MiningStats getStats() const;

private:
    std::atomic<bool> stop_mining_{false};

protected:
    // Configuration and device properties
    Config config_;
    gpuDeviceProp device_props_;
    GPUVendor gpu_vendor_;

    // GPU resources - using platform-independent types
    std::vector<DeviceMiningJob> device_jobs_;
    std::vector<gpuStream_t> streams_;
    std::vector<ResultPool> gpu_pools_;
    std::vector<MiningResult *> pinned_results_;
    std::vector<gpuEvent_t> start_events_;
    std::vector<gpuEvent_t> end_events_;

    // Performance tracking
    std::atomic<uint64_t> total_hashes_{0};
    std::atomic<uint64_t> total_candidates_{0};
    std::chrono::steady_clock::time_point start_time_;
    TimingStats timing_stats_;
    mutable std::mutex timing_mutex_;

    // Best result tracking
    BestResultTracker best_tracker_;

    // Thread management
    std::unique_ptr<std::thread> monitor_thread_;
    mutable std::mutex system_mutex_;

    // Callback management
    mutable std::mutex callback_mutex_;
    MiningResultCallback result_callback_;

    // Result accumulation
    mutable std::mutex all_results_mutex_;
    std::vector<MiningResult> all_results_;

    // Private methods
    bool initializeGPUResources();

    void cleanup();

    virtual void processResultsOptimized(int stream_idx);

    void performanceMonitor();

    void printFinalStats();

    uint64_t getTotalThreads() const;

    uint64_t getHashesPerKernel() const;

    // Platform detection and optimization
    GPUVendor detectGPUVendor() const;

    void optimizeForGPU();

    void autoTuneParameters();
};

// Declare the global mining system pointer
extern std::unique_ptr<MiningSystem> g_mining_system;

#endif // MINING_SYSTEM_HPP
