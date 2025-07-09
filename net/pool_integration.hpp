#pragma once

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <optional>
#include <deque>
#include <unordered_map>

#include "pool_protocol.hpp"
#include "pool_client.hpp"
#include "../multi_gpu_manager.hpp"

namespace MiningPool {
    /**
     * Integrated mining system that connects to a pool and manages GPU mining
     */
    class PoolMiningSystem : public IPoolEventHandler {
    public:
        std::atomic<uint64_t> job_generation_{0};
        std::atomic<bool> should_restart_mining_{false};

        /**
         * Configuration for pool mining
         */
        struct Config {
            // Pool connection settings
            PoolConfig pool_config;

            // GPU configuration
            int gpu_id = 0; // Single GPU ID (if not using multi-GPU)
            std::vector<int> gpu_ids; // Multiple GPU IDs (for multi-GPU)
            bool use_all_gpus = false; // Use all available GPUs

            // Mining settings
            uint32_t min_share_difficulty = 20; // Minimum difficulty to submit
            uint32_t share_scan_interval_ms = 100; // How often to scan for shares
            uint32_t stats_report_interval_s = 60; // How often to report stats
            bool enable_vardiff = true; // Enable variable difficulty
            uint32_t target_share_time = 30; // Target seconds between shares

            // Performance settings
            int mining_threads = 1; // Number of mining threads
            size_t result_buffer_size = 1024; // Size of result buffer
        };

        /**
         * Mining statistics
         */
        struct PoolMiningStats {
            // Connection status
            bool connected = false;
            bool authenticated = false;
            std::string worker_id;
            std::string pool_name;

            // Share statistics
            uint64_t shares_submitted = 0;
            uint64_t shares_accepted = 0;
            uint64_t shares_rejected = 0;
            double share_success_rate = 0.0;

            // Performance statistics
            double hashrate = 0.0;
            uint64_t total_hashes = 0;
            uint32_t current_difficulty = 0;

            // Timing
            std::chrono::seconds uptime{0};
            std::chrono::steady_clock::time_point last_share_time;
            std::chrono::steady_clock::time_point last_accepted_share_time;
        };

        // Constructor and destructor
        explicit PoolMiningSystem(const Config &config);

        ~PoolMiningSystem();

        // Control methods
        bool start();

        void stop();

        bool is_running() const { return running_.load(); }

        // Statistics
        PoolMiningStats get_stats() const;

        // IPoolEventHandler interface
        void on_connected() override;

        void on_disconnected(const std::string &reason) override;

        void on_error(ErrorCode code, const std::string &message) override;

        void on_authenticated(const std::string &worker_id) override;

        void on_auth_failed(ErrorCode code, const std::string &reason) override;

        void on_new_job(const PoolJob &job) override;

        void on_job_cancelled(const std::string &job_id) override;

        void on_share_accepted(const ShareResultMessage &result) override;

        void on_share_rejected(const ShareResultMessage &result) override;

        void on_difficulty_changed(uint32_t new_difficulty) override;

        void on_pool_status(const PoolStatusMessage &status) override;
    private:
        // Configuration
        Config config_;

        // Pool client
        std::unique_ptr<PoolClient> pool_client_;

        // Mining system (single or multi-GPU)
        std::unique_ptr<MiningSystem> mining_system_;
        std::unique_ptr<MultiGPUManager> multi_gpu_manager_;

        // Thread management
        std::atomic<bool> running_{false};
        std::atomic<bool> mining_active_{false};
        std::thread mining_thread_;
        std::thread share_scanner_thread_;
        std::thread stats_reporter_thread_;
        std::thread share_submission_thread_;

        // Job management
        mutable std::mutex job_mutex_;
        std::condition_variable job_cv_;
        std::optional<PoolJob> current_job_;
        std::optional<MiningJob> current_mining_job_;
        std::string current_job_id_for_mining_;
        std::atomic<uint32_t> current_difficulty_{20};

        // NEW: Track active jobs for proper validation
        std::unordered_map<std::string, PoolJob> active_jobs_;

        // NEW: Mining batch tracking
        struct MiningBatch {
            std::string job_id;
            MiningJob mining_job;
            std::chrono::steady_clock::time_point start_time;
        };

        std::mutex batch_mutex_;
        std::deque<MiningBatch> active_batches_;
        static constexpr size_t MAX_ACTIVE_BATCHES = 10;

        struct MiningResultWithJob {
            MiningResult result;
            std::string job_id;
        };

        // Result management
        mutable std::mutex results_mutex_;
        std::condition_variable results_cv_;
        std::vector<MiningResultWithJob> current_mining_results_with_job_;

        // Share submission queue
        std::mutex share_mutex_;
        std::condition_variable share_cv_;
        std::queue<Share> share_queue_;

        // Statistics
        mutable std::mutex stats_mutex_;
        PoolMiningStats stats_;
        std::chrono::steady_clock::time_point start_time_;
        std::deque<std::chrono::steady_clock::time_point> share_times_;

        // Internal methods
        void mining_loop();

        void share_scanner_loop();

        void stats_reporter_loop();

        void share_submission_loop();

        void setup_mining_result_callback();

        void process_mining_results(const std::vector<MiningResult> &results, const std::string &job_id);

        void scan_for_shares();

        void submit_share(const MiningResult &result);

        void submit_share_direct(const Share &share);

        MiningJob convert_to_mining_job(const JobMessage &job_msg);

        void update_mining_job(const PoolJob &pool_job);

        void update_stats();

        void handle_reconnect();

        void cleanup_mining_system();

        void adjust_local_difficulty();

        uint32_t calculate_optimal_scan_difficulty();
    };

    /**
     * Manager for multiple pool connections with failover support
     */
    class MultiPoolManager {
    public:
        MultiPoolManager();

        ~MultiPoolManager();

        // Pool management
        void add_pool(const std::string &name, const PoolConfig &config, int priority = 0);

        void remove_pool(const std::string &name);

        void set_pool_priority(const std::string &name, int priority);

        void enable_pool(const std::string &name, bool enable);

        // Mining control
        bool start_mining(const PoolMiningSystem::Config &mining_config);

        void stop_mining();

        // Status
        std::string get_active_pool() const;

        std::map<std::string, PoolMiningSystem::PoolMiningStats> get_all_stats() const;

    private:
        struct PoolEntry {
            std::string name;
            PoolConfig config;
            int priority;
            bool enabled;
            std::unique_ptr<PoolMiningSystem> mining_system;
        };

        mutable std::mutex mutex_;
        std::vector<PoolEntry> pools_;
        std::string active_pool_;
        std::atomic<bool> running_{false};

        PoolMiningSystem::Config base_mining_config_;
        std::thread failover_thread_;

        void failover_monitor();

        bool try_next_pool();

        void sort_pools_by_priority();
    };
} // namespace MiningPool
