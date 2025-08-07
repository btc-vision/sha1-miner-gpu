#ifndef MULTI_GPU_MANAGER_HPP
#define MULTI_GPU_MANAGER_HPP

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "sha1_miner.cuh"

#include "../src/core/mining_system.hpp"

// Callback type for multi-GPU results
using MiningResultCallback = std::function<void(const std::vector<MiningResult> &)>;

// Global batch size for nonce distribution across GPUs
constexpr uint64_t NONCE_BATCH_SIZE = 1ULL << 32;  // 4B nonces per batch

struct GPUWorker
{
    int device_id;
    std::unique_ptr<MiningSystem> mining_system;
    std::unique_ptr<std::thread> worker_thread;
    std::atomic<uint64_t> hashes_computed{0};
    std::atomic<uint64_t> candidates_found{0};
    std::atomic<uint32_t> best_match_bits{0};
    std::atomic<bool> active{false};
};

/**
 * Multi-GPU mining manager that coordinates mining across multiple GPUs
 * Each GPU runs independently with its own nonce range
 */
class MultiGPUManager final
{
public:
    MultiGPUManager();
    ~MultiGPUManager();

    void setUserConfig(const void *user_config) { user_config_ = user_config; }

    /**
     * Initialize mining on specified GPUs
     * @param gpu_ids List of GPU device IDs to use
     * @return true if at least one GPU was initialized successfully
     */
    bool initialize(const std::vector<int> &gpu_ids);

    /**
     * Set a callback to be called whenever any GPU finds results
     * @param callback Function to call with new results
     */
    void setResultCallback(const MiningResultCallback &callback)
    {
        std::lock_guard lock(callback_mutex_);
        result_callback_ = callback;
    }

    /**
     * Run mining on all initialized GPUs (infinite mode)
     * @param job Mining job configuration
     */
    void runMining(const MiningJob &job);

    void sync() const;

    /**
     * Run mining with interruption capability
     * @param job Mining job configuration
     * @param should_continue Function that returns false when mining should stop
     * @param global_nonce_offset Atomic nonce offset shared across all GPUs
     */
    void runMiningInterruptibleWithOffset(const MiningJob &job, std::function<bool()> should_continue,
                                          std::atomic<uint64_t> &global_nonce_offset);

    /**
     * Stop all mining operations
     */
    void stopMining();

    /**
     * Update job on all GPUs without stopping mining
     * @param job New mining job
     * @param job_version Version identifier for the new job
     */
    void updateJobLive(const MiningJob &job, uint64_t job_version) const;

    /**
     * Get combined statistics from all GPUs
     */
    // void printCombinedStats() const;

    /**
     * Get total hash rate across all GPUs
     */
    double getTotalHashRate() const;

    /**
     * Get number of active workers
     */
    size_t getActiveWorkerCount() const;

private:
    const void *user_config_ = nullptr;

    // Worker management
    std::vector<std::unique_ptr<GPUWorker>> workers_;
    std::atomic<bool> shutdown_{false};

    // Global nonce distribution
    std::atomic<uint64_t> global_nonce_counter_{1};

    // Performance tracking
    mutable std::chrono::steady_clock::time_point start_time_;
    mutable BestResultTracker global_best_tracker_;
    mutable std::mutex stats_mutex_;
    uint32_t current_difficulty_{0};

    // Callback management
    mutable std::mutex callback_mutex_;
    MiningResultCallback result_callback_;

    // Monitor thread
    // std::unique_ptr<std::thread> monitor_thread_;

    // Worker thread functions
    void workerThread(GPUWorker *worker, const MiningJob &job);
    void workerThreadInterruptibleWithOffset(GPUWorker *worker, const MiningJob &job,
                                             std::function<bool()> should_continue,
                                             std::atomic<uint64_t> &shared_nonce_counter);

    // Get next batch of nonces for a worker
    uint64_t getNextNonceBatch();

    // Process results from a worker
    void processWorkerResults(GPUWorker *worker, const std::vector<MiningResult> &results);

    // Check if all workers are ready
    bool allWorkersReady() const;

    // Wait for workers to finish
    void waitForWorkers() const;
    void waitForWorkersWithTimeout(std::chrono::seconds timeout) const;
};

#endif  // MULTI_GPU_MANAGER_HPP
