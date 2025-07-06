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
        int device_id = 0;
        int num_streams = 4;
        int blocks_per_stream = 0; // Auto-calculate
        int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
        bool use_pinned_memory = true;
        size_t result_buffer_size = MAX_CANDIDATES_PER_BATCH;
        bool force_generic_kernel = false; // Force use of generic kernel
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

    MiningSystem(const Config &config = {});

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

// C-style interface functions
#ifdef __cplusplus
extern "C" {
#endif

bool init_mining_system(int device_id);

void cleanup_mining_system();

MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

void run_mining_loop(MiningJob job, uint32_t duration_seconds);

#ifdef __cplusplus
}
#endif

#endif // MINING_SYSTEM_HPP
