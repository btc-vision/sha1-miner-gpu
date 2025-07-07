#ifndef MULTI_GPU_MANAGER_HPP
#define MULTI_GPU_MANAGER_HPP

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>
#include "../src/mining_system.hpp"

/**
 * Manages mining across multiple GPUs
 * Each GPU runs in its own thread with independent nonce ranges
 */
class MultiGPUManager {
public:
    /**
     * Worker structure for each GPU
     */
    struct GPUWorker {
        int device_id;
        std::unique_ptr<MiningSystem> mining_system;
        std::unique_ptr<std::thread> worker_thread;
        std::atomic<uint64_t> hashes_computed{0};
        std::atomic<uint64_t> candidates_found{0};
        std::atomic<uint32_t> best_match_bits{0};
    };

    MultiGPUManager();

    ~MultiGPUManager();

    /**
     * Initialize the manager with specified GPU IDs
     * @param gpu_ids List of GPU device IDs to use
     * @return true if at least one GPU was initialized successfully
     */
    bool initialize(const std::vector<int> &gpu_ids);

    /**
     * Run mining on all initialized GPUs
     * @param job Mining job to execute
     * @param duration_seconds How long to mine
     */
    void runMining(const MiningJob &job, uint32_t duration_seconds);

    /**
     * Print combined statistics from all GPUs
     */
    void printCombinedStats();

    /**
     * Get the number of active GPUs
     */
    size_t getNumGPUs() const { return workers_.size(); }

private:
    std::vector<std::unique_ptr<GPUWorker> > workers_;
    std::mutex stats_mutex_;
    std::atomic<bool> shutdown_{false};
    std::chrono::steady_clock::time_point start_time_;

    // Global best result tracking
    BestResultTracker global_best_tracker_;

    // Nonce distribution
    std::atomic<uint64_t> global_nonce_counter_{1};

    // Each GPU gets a batch of 2^40 nonces (about 1 trillion)
    // This ensures GPUs don't overlap even in very long runs
    static constexpr uint64_t NONCE_BATCH_SIZE = 1ULL << 40;

    // Store job difficulty for stats calculation
    uint32_t current_difficulty_ = 0;

    /**
     * Worker thread function for each GPU
     */
    void workerThread(GPUWorker *worker, const MiningJob &job, uint32_t duration_seconds);

    /**
     * Get the next batch of nonces for a GPU
     */
    uint64_t getNextNonceBatch();
};

#endif // MULTI_GPU_MANAGER_HPP
