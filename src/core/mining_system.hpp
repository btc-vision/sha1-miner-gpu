#ifndef MINING_SYSTEM_HPP
#define MINING_SYSTEM_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "sha1_miner.cuh"

#include "../../logging/logger.hpp"
#include "gpu_platform.hpp"

#ifdef USE_HIP
    #include "architecture/gpu_architecture.hpp"
#endif

// Forward declare the global shutdown flag
extern std::atomic<bool> g_shutdown;

/**
 * Callback type for processing mining results in real-time
 */
using MiningResultCallback = std::function<void(const std::vector<MiningResult> &)>;

/**
 * Thread-safe tracker for best mining results
 */
class BestResultTracker
{
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
enum class GPUVendor { NVIDIA, AMD, UNKNOWN };

/**
 * Enhanced mining system with proper resource management
 * Supports both NVIDIA and AMD GPUs
 */
class MiningSystem final
{
public:
    struct Config
    {
        int device_id;
        int num_streams;
        int blocks_per_stream;
        int threads_per_block;
        bool use_pinned_memory;
        size_t result_buffer_size;

        // Constructor with default values
        Config()
            : device_id(0),
              num_streams(0),
              blocks_per_stream(0),
              threads_per_block(0),
              use_pinned_memory(true),
              result_buffer_size(0)
        {}
    };

    void sync() const;

    bool validateStreams();

    /**
     * Stop all mining operations
     */
    void stopMining();

    /**
     * Set a callback to be called whenever new results are found
     * @param callback Function to call with new results
     */
    void setResultCallback(MiningResultCallback callback)
    {
        std::lock_guard lock(callback_mutex_);
        result_callback_ = callback;
    }

    /**
     * Run a single batch of mining without the monitoring thread
     * Used by MultiGPUManager for proper hash tracking
     * @return Number of hashes computed in this batch
     */
    uint64_t runSingleBatch(const MiningJob &job);

    /**
     * Get results from the last batch
     * @return Vector of mining results
     */
    std::vector<MiningResult> getLastResults() const
    {
        std::vector<MiningResult> results;
        // Get result count from first pool
        uint32_t count;
        (void)gpuMemcpy(&count, gpu_pools_[0].count, sizeof(uint32_t), gpuMemcpyDeviceToHost);
        if (count > 0 && count <= gpu_pools_[0].capacity) {
            results.resize(count);
            (void)gpuMemcpy(results.data(), gpu_pools_[0].results, sizeof(MiningResult) * count, gpuMemcpyDeviceToHost);
        }
        return results;
    }

    void setUserConfig(const void *user_config) { user_config_ = user_config; }

    /**
     * Get current configuration
     */
    const Config &getConfig() const { return config_; }

    /**
     * Reset internal state for new mining session
     */
    void resetState();

    /**
     * Get all results found since last clear
     * Used for batch processing
     */
    std::vector<MiningResult> getAllResults()
    {
        std::lock_guard lock(all_results_mutex_);
        auto results = all_results_;
        all_results_.clear();
        return results;
    }

    /**
     * Clear all accumulated results
     */
    void clearResults();

    // Timing statistics structure
    struct TimingStats
    {
        double kernel_launch_time_ms    = 0;
        double kernel_execution_time_ms = 0;
        double result_copy_time_ms      = 0;
        double total_kernel_time_ms     = 0;
        int kernel_count                = 0;

        void reset();

        void print() const;
    };

    // Constructor with default config
    explicit MiningSystem(const Config &config = Config());

    virtual ~MiningSystem();

    /**
     * Initialize the mining system
     * @return true if successful, false otherwise
     */
    bool initialize();

    /**
     * Run infinite mining loop
     * @param job Mining job configuration
     */
    void runMiningLoop(const MiningJob &job);

    /**
     * Run interruptible mining loop
     * @param job Mining job configuration
     * @param should_continue Function that returns false when mining should stop
     */
    uint64_t runMiningLoopInterruptibleWithOffset(const MiningJob &job, const std::function<bool()> &should_continue,
                                                  uint64_t start_nonce);

    /**
     * Get current mining statistics
     */
    MiningStats getStats() const;

    /**
     * Update mining job without stopping (live update)
     * @param job New mining job
     * @param job_version Version identifier for the new job
     */
    void updateJobLive(const MiningJob &job, uint64_t job_version);

private:
    const void *user_config_ = nullptr;

    struct UserSpecifiedFlags
    {
        bool threads = false;
        bool streams = false;
        bool blocks  = false;
        bool buffer  = false;
    };

    struct OptimalConfig
    {
        int threads;
        int streams;
        int blocks_per_sm;
        size_t buffer_size;
    };

    UserSpecifiedFlags detectUserSpecifiedValues() const;
    OptimalConfig determineOptimalConfig();
    OptimalConfig getAMDOptimalConfig();
    OptimalConfig getNVIDIAOptimalConfig() const;

    void applyUserSpecifiedValues(const UserSpecifiedFlags &user_flags, const OptimalConfig &optimal);
    void validateConfiguration();
    void adjustForMemoryConstraints(const UserSpecifiedFlags &user_flags);
    void logFinalConfiguration(const UserSpecifiedFlags &user_flags);

    std::atomic<bool> stop_mining_{false};

    // Event-based synchronization
    std::vector<gpuEvent_t> kernel_complete_events_;
    std::vector<std::chrono::high_resolution_clock::time_point> kernel_launch_times_;

    // Reduce CPU usage with better scheduling
    std::condition_variable work_cv_;
    std::mutex work_mutex_;
    std::atomic<int> active_streams_{0};

    // Structure to track per-stream state
    struct StreamData
    {
        uint64_t nonce_offset;
        bool busy;
        std::chrono::high_resolution_clock::time_point launch_time;
        uint64_t last_nonces_processed;
    };

    std::atomic<uint64_t> current_job_version_{0};

    // Helper methods
    void launchKernelOnStream(int stream_idx, uint64_t nonce_offset, const MiningJob &job);

    void processStreamResults(int stream_idx, StreamData &stream_data);

    void performanceMonitorInterruptible(const std::function<bool()> &should_continue) const;

protected:
    // Configuration and device properties
    Config config_;
    gpuDeviceProp device_props_;
    GPUVendor gpu_vendor_;

#ifdef USE_HIP
    AMDArchitecture detected_arch_ = AMDArchitecture::UNKNOWN;
#endif

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
    // std::unique_ptr<std::thread> monitor_thread_;
    mutable std::mutex system_mutex_;

    // Callback management
    mutable std::mutex callback_mutex_;
    MiningResultCallback result_callback_;

    // Result accumulation
    mutable std::mutex all_results_mutex_;
    std::vector<MiningResult> all_results_;

    // Private methods
    bool initializeGPUResources();

    bool initializeMemoryPools();

    void cleanup();

    void processResultsOptimized(int stream_idx);

    void performanceMonitor() const;

    void printFinalStats() const;

    uint64_t getTotalThreads() const;

    uint64_t getHashesPerKernel() const;

    // Platform detection and optimization
    GPUVendor detectGPUVendor() const;

    static void optimizeForGPU();

    void autoTuneParameters();
};

// Declare the global mining system pointer
extern std::unique_ptr<MiningSystem> g_mining_system;

#endif  // MINING_SYSTEM_HPP
