#pragma once

#include "pool_client.hpp"
#include "../src/mining_system.hpp"
#include "../include/multi_gpu_manager.hpp"
#include "sha1_miner.cuh"
#include <memory>
#include <thread>
#include <queue>
#include <deque>
#include <condition_variable>

namespace MiningPool {
    static constexpr size_t MAX_SHARE_QUEUE_SIZE = 10000;
    static constexpr size_t MAX_PENDING_RESULTS = 50000;

    // Pool-aware mining system that integrates with the existing miner
    class PoolMiningSystem : public IPoolEventHandler {
    public:
        struct Config {
            PoolConfig pool_config;

            // Mining settings
            bool use_all_gpus = false;
            std::vector<int> gpu_ids;
            int gpu_id = 0;

            // Pool settings
            uint32_t min_share_difficulty = 20;   // Minimum difficulty in BITS (default: 20 bits)
            uint32_t max_share_difficulty = 55;   // Maximum difficulty in BITS (default: 50 bits)
            bool enable_vardiff = true;
            double target_share_time = 30.0;      // Target seconds between shares
            uint32_t share_scan_interval_ms = 100;
        };

        PoolMiningSystem(const Config &config);

        ~PoolMiningSystem();

        // Start/stop pool mining
        bool start();

        void share_submission_loop();

        void stop();

        bool is_running() const { return running_.load(); }

        struct PoolMiningStats {
            // Connection
            bool connected = false;
            bool authenticated = false;
            std::string worker_id;
            std::string pool_name;

            // Mining
            uint64_t total_hashes = 0;
            double hashrate = 0.0;

            // Shares
            uint64_t shares_submitted = 0;
            uint64_t shares_accepted = 0;
            uint64_t shares_rejected = 0;
            double share_success_rate = 0.0;

            // Difficulty - NOW TRACKING BITS
            uint32_t current_difficulty = 0;      // Current difficulty in BITS
            uint32_t current_bits = 0;            // Explicit bit tracking (same as current_difficulty)
            uint32_t best_share_bits = 0;         // Best share found in BITS
            double total_scaled_difficulty = 0;   // Total difficulty as sum of 2^bits
            double total_difficulty_accepted = 0; // Backward compatibility

            // Timing
            std::chrono::seconds uptime{0};
        };

        PoolMiningStats get_stats() const;

        // IPoolEventHandler implementation
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

        void process_mining_results(const std::vector<MiningResult> &results);

        MiningJob convert_to_mining_job(const JobMessage &job_msg);

    private:
        std::vector<MiningResult> pending_results_;
        std::mutex pending_results_mutex_;

        // ADD: Better job synchronization
        struct JobUpdate {
            PoolJob pool_job;
            MiningJob mining_job;
            uint64_t version;
            std::string job_id;
        };
        std::atomic<JobUpdate*> pending_job_update_{nullptr};

        std::atomic<uint64_t> job_version_{0};
        std::atomic<bool> job_update_pending_{false};

        void update_mining_job_live(const PoolJob &pool_job);

        std::atomic<uint32_t> current_difficulty_{20};

        Config config_;
        std::unique_ptr<PoolClient> pool_client_;
        std::unique_ptr<MiningSystem> mining_system_;
        std::unique_ptr<MultiGPUManager> multi_gpu_manager_;

        // State
        std::atomic<bool> running_{false};
        std::atomic<bool> mining_active_{false};

        // Current job
        mutable std::mutex job_mutex_;
        std::condition_variable job_cv_;
        std::optional<PoolJob> current_job_;
        std::optional<MiningJob> current_mining_job_;
        std::string current_job_id_for_mining_;

        // Share management
        std::mutex share_mutex_;
        std::condition_variable share_cv_;
        std::queue<Share> share_queue_;
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_shares_;

        // Result accumulation for proper integration
        std::mutex results_mutex_;
        std::condition_variable results_cv_;
        std::vector<MiningResult> current_mining_results_;

        // Share timing for vardiff
        std::deque<std::chrono::steady_clock::time_point> share_times_;

        // Statistics
        mutable std::mutex stats_mutex_;
        PoolMiningStats stats_;
        std::chrono::steady_clock::time_point start_time_;

        std::thread share_submission_thread_;

        // Threads
        std::thread mining_thread_;
        std::thread share_scanner_thread_;
        std::thread stats_reporter_thread_;

        // Internal methods
        void mining_loop();

        void share_scanner_loop();

        void stats_reporter_loop();

        // Job management
        void update_mining_job(const PoolJob &pool_job);

        // Share processing
        void scan_for_shares();

        void setup_mining_result_callback();

        void submit_share(const MiningResult &result);

        // Vardiff
        void adjust_local_difficulty();

        uint32_t calculate_optimal_scan_difficulty();

        // Utilities
        void update_stats();

        void handle_reconnect();

        void cleanup_mining_system();
    };

    // Multi-pool manager with failover
    class MultiPoolManager {
    public:
        struct PoolEntry {
            std::string name;
            PoolConfig config;
            int priority; // Lower = higher priority
            bool enabled;
            std::unique_ptr<PoolMiningSystem> mining_system;
        };

        MultiPoolManager();

        ~MultiPoolManager();

        // Pool management
        void add_pool(const std::string &name, const PoolConfig &config, int priority = 0);

        void remove_pool(const std::string &name);

        void set_pool_priority(const std::string &name, int priority);

        void enable_pool(const std::string &name, bool enable);

        // Start mining with automatic failover
        bool start_mining(const PoolMiningSystem::Config &mining_config);

        void stop_mining();

        // Get current active pool
        std::string get_active_pool() const;

        // Get statistics for all pools
        std::map<std::string, PoolMiningSystem::PoolMiningStats> get_all_stats() const;

    private:
        mutable std::mutex mutex_;
        std::vector<PoolEntry> pools_;
        std::string active_pool_;
        PoolMiningSystem::Config base_mining_config_;

        std::thread failover_thread_;
        std::atomic<bool> running_{false};

        void failover_monitor();

        bool try_next_pool();

        void sort_pools_by_priority();
    };

    // Example pool configurations
    namespace PoolPresets {
        inline PoolConfig create_default_pool(const std::string &url,
                                              const std::string &wallet,
                                              const std::string &worker_name) {
            PoolConfig config;
            config.url = url;
            config.username = wallet;
            config.worker_name = worker_name;
            config.auth_method = AuthMethod::WORKER_PASS;

            // Determine if TLS from URL
            config.use_tls = (url.find("wss://") == 0);

            // Connection settings
            config.keepalive_interval_s = 30;
            config.response_timeout_ms = 10000;
            config.reconnect_delay_ms = 5000;
            config.max_reconnect_delay_ms = 60000;

            return config;
        }

        // Common pool configurations
        inline PoolConfig create_local_pool(const std::string &worker_name) {
            return create_default_pool("ws://localhost:3333", "local", worker_name);
        }

        inline PoolConfig create_ssl_pool(const std::string &url,
                                          const std::string &wallet,
                                          const std::string &worker_name,
                                          const std::string &cert_file = "",
                                          const std::string &key_file = "") {
            auto config = create_default_pool(url, wallet, worker_name);
            config.use_tls = true;
            config.verify_server_cert = true;

            if (!cert_file.empty()) {
                config.tls_cert_file = cert_file;
                config.tls_key_file = key_file;
                config.auth_method = AuthMethod::CERTIFICATE;
            }

            return config;
        }
    }
} // namespace MiningPool
