#ifndef MULTI_GPU_MANAGER_HPP
#define MULTI_GPU_MANAGER_HPP

#include <vector>
#include <thread>
#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>
#include <functional>
#include "sha1_miner.cuh"
#include "../src/mining_system.hpp"

// Callback type for multi-GPU results
using MiningResultCallback = std::function<void(const std::vector<MiningResult> &)>;

// Global batch size for nonce distribution across GPUs
constexpr uint64_t NONCE_BATCH_SIZE = 1ULL << 32; // 4B nonces per batch

/**
 * Multi-GPU mining manager that coordinates mining across multiple GPUs
 * Each GPU runs independently with its own nonce range
 */
class MultiGPUManager {
public:
    MultiGPUManager();

    ~MultiGPUManager();

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
    void setResultCallback(MiningResultCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        result_callback_ = callback;
    }

    /**
     * Run mining on all initialized GPUs
     * @param job Mining job configuration
     * @param duration_seconds How long to mine
     */
    void runMining(const MiningJob &job, uint32_t duration_seconds);

    /**
     * Get combined statistics from all GPUs
     */
    void printCombinedStats();

protected:
    struct GPUWorker {
        int device_id;
        std::unique_ptr<MiningSystem> mining_system;
        std::unique_ptr<std::thread> worker_thread;
        std::atomic<uint64_t> hashes_computed{0};
        std::atomic<uint64_t> candidates_found{0};
        std::atomic<uint32_t> best_match_bits{0};
    };

    std::vector<std::unique_ptr<GPUWorker> > workers_;
    std::atomic<bool> shutdown_{false};

    // Global nonce distribution
    std::atomic<uint64_t> global_nonce_counter_{1};

    // Performance tracking
    std::chrono::steady_clock::time_point start_time_;
    BestResultTracker global_best_tracker_;
    std::mutex stats_mutex_;
    uint32_t current_difficulty_;

    // Callback management
    mutable std::mutex callback_mutex_;
    MiningResultCallback result_callback_;

    // Worker thread function
    virtual void workerThread(GPUWorker *worker, const MiningJob &job, uint32_t duration_seconds);

    // Get next batch of nonces for a worker
    uint64_t getNextNonceBatch();
};

#endif // MULTI_GPU_MANAGER_HPP
