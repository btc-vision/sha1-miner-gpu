#ifndef MINING_SYSTEM_HPP
#define MINING_SYSTEM_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>

#include "gpu_platform.hpp"
#include "sha1_miner.cuh"

// Forward declare the global shutdown flag
extern std::atomic<bool> g_shutdown;

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

    /**
     * Reset internal state for a new mining run
     * Used by multi-GPU manager between rounds
     */
    void resetState() {
        total_hashes_ = 0;
        total_candidates_ = 0;
        best_tracker_.reset();
        start_time_ = std::chrono::steady_clock::now();
        timing_stats_.reset();
    }

    // Constructor with default config
    explicit MiningSystem(const Config &config = Config());

    ~MiningSystem();

    bool initialize();

    void runMiningLoop(const MiningJob &job, uint32_t duration_seconds);

    MiningStats getStats() const;

private:
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

    // Private methods
    bool initializeGPUResources();

    void cleanup();

    void processResultsOptimized(int stream_idx);

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
