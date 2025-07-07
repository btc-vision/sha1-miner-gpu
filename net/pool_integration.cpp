// pool_integration.cpp - Integration of pool client with mining system
#include "pool_integration.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

namespace MiningPool {
    // PoolMiningSystem implementation
    PoolMiningSystem::PoolMiningSystem(const Config &config)
        : config_(config) {
        // Initialize statistics
        stats_ = {};
        stats_.current_difficulty = config_.min_share_difficulty;
        start_time_ = std::chrono::steady_clock::now();
    }

    PoolMiningSystem::~PoolMiningSystem() {
        stop();
    }

    bool PoolMiningSystem::start() {
        if (running_.load()) {
            return true;
        }

        std::cout << "Starting pool mining system..." << std::endl;

        // Create pool client
        pool_client_ = std::make_unique<PoolClient>(config_.pool_config, this);

        // Initialize mining system
        MiningSystem::Config mining_config;
        if (config_.use_all_gpus) {
            // Use all available GPUs
            int device_count;
            gpuGetDeviceCount(&device_count);
            mining_config.device_id = 0; // Primary GPU
            // Note: For multi-GPU, we'd need to create multiple MiningSystem instances
        } else if (!config_.gpu_ids.empty()) {
            mining_config.device_id = config_.gpu_ids[0]; // Use first GPU for now
        } else {
            mining_config.device_id = config_.gpu_id;
        }

        mining_system_ = std::make_unique<MiningSystem>(mining_config);
        if (!mining_system_->initialize()) {
            std::cerr << "Failed to initialize mining system" << std::endl;
            return false;
        }

        // Connect to pool
        if (!pool_client_->connect()) {
            std::cerr << "Failed to connect to pool" << std::endl;
            return false;
        }

        running_ = true;

        // Start worker threads
        mining_thread_ = std::thread(&PoolMiningSystem::mining_loop, this);
        share_scanner_thread_ = std::thread(&PoolMiningSystem::share_scanner_loop, this);
        stats_reporter_thread_ = std::thread(&PoolMiningSystem::stats_reporter_loop, this);

        return true;
    }

    void PoolMiningSystem::stop() {
        if (!running_.load()) {
            return;
        }

        std::cout << "Stopping pool mining system..." << std::endl;

        running_ = false;
        mining_active_ = false;

        // Disconnect from pool
        if (pool_client_) {
            pool_client_->disconnect();
        }

        // Stop mining
        if (mining_system_) {
            cleanup_mining_system();
        }

        // Join threads
        if (mining_thread_.joinable()) {
            mining_thread_.join();
        }
        if (share_scanner_thread_.joinable()) {
            share_scanner_thread_.join();
        }
        if (stats_reporter_thread_.joinable()) {
            stats_reporter_thread_.join();
        }

        std::cout << "Pool mining system stopped" << std::endl;
    }

    void PoolMiningSystem::mining_loop() {
        while (running_.load()) {
            if (!mining_active_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            std::unique_lock<std::mutex> lock(job_mutex_);
            if (!current_mining_job_.has_value()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            auto job = current_mining_job_.value();
            lock.unlock();

            // Run mining for a short period
            const uint32_t mining_duration = 30; // 30 seconds per batch
            mining_system_->runMiningLoop(job, mining_duration);

            // The share scanner will pick up any results
        }
    }

    void PoolMiningSystem::share_scanner_loop() {
        while (running_.load()) {
            scan_for_shares();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.share_scan_interval_ms)
            );
        }
    }

    void PoolMiningSystem::stats_reporter_loop() {
        while (running_.load()) {
            // Report hashrate every 30 seconds
            std::this_thread::sleep_for(std::chrono::seconds(30));

            if (!pool_client_->is_connected() || !mining_active_.load()) {
                continue;
            }

            auto mining_stats = mining_system_->getStats();

            HashrateReportMessage report;
            report.hashrate = mining_stats.hash_rate;
            report.gpu_count = config_.use_all_gpus ? config_.gpu_ids.size() : 1;
            report.shares_found = stats_.shares_submitted;
            report.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_
            ).count();

            pool_client_->report_hashrate(report);

            // Update local stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.hashrate = mining_stats.hash_rate;
                stats_.total_hashes = mining_stats.hashes_computed;
            }
        }
    }

    void PoolMiningSystem::scan_for_shares() {
        if (!mining_system_ || !mining_active_.load()) {
            return;
        }

        // Get current mining stats
        auto mining_stats = mining_system_->getStats();

        // Check if we have any candidates that meet pool difficulty
        uint32_t pool_difficulty = current_difficulty_.load();

        // In a real implementation, we'd need to access the mining system's
        // result buffer directly. For now, we'll check based on the best match
        if (mining_stats.best_match_bits >= pool_difficulty) {
            // We found a share!
            // In practice, we'd need to get the actual nonce and hash
            MiningResult result;
            result.matching_bits = mining_stats.best_match_bits;
            // result.nonce would be set from actual mining results
            // result.hash would be set from actual mining results

            submit_share(result);
        }
    }

    void PoolMiningSystem::submit_share(const MiningResult &result) {
        std::lock_guard<std::mutex> lock(job_mutex_);
        if (!current_job_.has_value()) {
            return;
        }

        Share share;
        share.job_id = current_job_->job_id;
        share.nonce = result.nonce;
        share.hash.resize(20);
        // Convert uint32_t[5] to uint8_t[20]
        for (int i = 0; i < 5; i++) {
            share.hash[i * 4 + 0] = (result.hash[i] >> 24) & 0xFF;
            share.hash[i * 4 + 1] = (result.hash[i] >> 16) & 0xFF;
            share.hash[i * 4 + 2] = (result.hash[i] >> 8) & 0xFF;
            share.hash[i * 4 + 3] = result.hash[i] & 0xFF;
        }
        share.matching_bits = result.matching_bits;
        share.found_time = std::chrono::steady_clock::now();

        // Submit to pool
        pool_client_->submit_share(share);

        // Track pending share
        {
            std::lock_guard<std::mutex> lock(shares_mutex_);
            PendingShare pending;
            pending.share = share;
            pending.submit_time = std::chrono::steady_clock::now();

            std::string share_id = share.job_id + "_" + std::to_string(share.nonce);
            pending_shares_[share_id] = pending;
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.shares_submitted++;
        }
    }

    MiningJob PoolMiningSystem::convert_to_mining_job(const JobMessage &job_msg) {
        MiningJob mining_job;

        // Copy base message
        if (job_msg.base_message.size() == 32) {
            std::copy(job_msg.base_message.begin(), job_msg.base_message.end(),
                      mining_job.base_message);
        }

        // Convert target hash from uint8_t[20] to uint32_t[5]
        if (job_msg.target_hash.size() == 20) {
            for (int i = 0; i < 5; i++) {
                mining_job.target_hash[i] =
                        (static_cast<uint32_t>(job_msg.target_hash[i * 4]) << 24) |
                        (static_cast<uint32_t>(job_msg.target_hash[i * 4 + 1]) << 16) |
                        (static_cast<uint32_t>(job_msg.target_hash[i * 4 + 2]) << 8) |
                        static_cast<uint32_t>(job_msg.target_hash[i * 4 + 3]);
            }
        }

        mining_job.difficulty = job_msg.difficulty;
        mining_job.nonce_offset = job_msg.nonce_start;

        return mining_job;
    }

    void PoolMiningSystem::update_mining_job(const PoolJob &pool_job) {
        std::lock_guard<std::mutex> lock(job_mutex_);

        current_job_ = pool_job;
        current_mining_job_ = convert_to_mining_job(pool_job.job_data);

        // Update difficulty
        current_difficulty_ = pool_job.job_data.difficulty;

        // Start mining if not already active
        mining_active_ = true;
    }

    PoolMiningSystem::PoolMiningStats PoolMiningSystem::get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        auto stats = stats_;

        // Update uptime
        stats.uptime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_
        );

        // Calculate success rate
        if (stats.shares_submitted > 0) {
            stats.share_success_rate = static_cast<double>(stats.shares_accepted) /
                                       static_cast<double>(stats.shares_submitted);
        }

        // Get connection status
        if (pool_client_) {
            stats.connected = pool_client_->is_connected();
            stats.authenticated = stats.connected && !stats.worker_id.empty();
        }

        return stats;
    }

    // IPoolEventHandler implementations
    void PoolMiningSystem::on_connected() {
        std::cout << "Connected to mining pool" << std::endl;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.connected = true;
    }

    void PoolMiningSystem::on_disconnected(const std::string &reason) {
        std::cout << "Disconnected from pool: " << reason << std::endl;

        mining_active_ = false;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.connected = false;
        stats_.authenticated = false;
    }

    void PoolMiningSystem::on_error(ErrorCode code, const std::string &message) {
        std::cerr << "Pool error (" << static_cast<int>(code) << "): " << message << std::endl;
    }

    void PoolMiningSystem::on_authenticated(const std::string &worker_id) {
        std::cout << "Authenticated as worker: " << worker_id << std::endl;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.authenticated = true;
        stats_.worker_id = worker_id;
    }

    void PoolMiningSystem::on_auth_failed(ErrorCode code, const std::string &reason) {
        std::cerr << "Authentication failed: " << reason << std::endl;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.authenticated = false;
    }

    void PoolMiningSystem::on_new_job(const PoolJob &job) {
        std::cout << "New job received: " << job.job_id
                << " (difficulty: " << job.job_data.difficulty << ")" << std::endl;

        update_mining_job(job);
    }

    void PoolMiningSystem::on_job_cancelled(const std::string &job_id) {
        std::cout << "Job cancelled: " << job_id << std::endl;

        std::lock_guard<std::mutex> lock(job_mutex_);
        if (current_job_.has_value() && current_job_->job_id == job_id) {
            current_job_.reset();
            current_mining_job_.reset();
            mining_active_ = false;
        }
    }

    void PoolMiningSystem::on_share_accepted(const ShareResultMessage &result) {
        std::cout << "Share accepted! Difficulty: " << result.difficulty_credited << std::endl;

        // Remove from pending
        {
            std::lock_guard<std::mutex> lock(shares_mutex_);
            std::string share_id = result.job_id + "_" + result.share_id;
            pending_shares_.erase(share_id);
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.shares_accepted++;
        }
    }

    void PoolMiningSystem::on_share_rejected(const ShareResultMessage &result) {
        std::cout << "Share rejected: " << result.message << std::endl;

        // Remove from pending
        {
            std::lock_guard<std::mutex> lock(shares_mutex_);
            std::string share_id = result.job_id + "_" + result.share_id;
            pending_shares_.erase(share_id);
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.shares_rejected++;
        }
    }

    void PoolMiningSystem::on_difficulty_changed(uint32_t new_difficulty) {
        std::cout << "Difficulty adjusted to: " << new_difficulty << std::endl;

        current_difficulty_ = new_difficulty;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.current_difficulty = new_difficulty;
    }

    void PoolMiningSystem::on_pool_status(const PoolStatusMessage &status) {
        std::cout << "Pool status - Workers: " << status.active_workers
                << ", Hashrate: " << status.pool_hashrate / 1e9 << " GH/s" << std::endl;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.pool_name = status.network_info["pool_name"];
    }

    // MultiPoolManager implementation
    MultiPoolManager::MultiPoolManager() = default;

    MultiPoolManager::~MultiPoolManager() {
        stop_mining();
    }

    void MultiPoolManager::add_pool(const std::string &name, const PoolConfig &config, int priority) {
        std::lock_guard<std::mutex> lock(mutex_);

        PoolEntry entry;
        entry.name = name;
        entry.config = config;
        entry.priority = priority;
        entry.enabled = true;

        pools_.push_back(std::move(entry));
        sort_pools_by_priority();
    }

    void MultiPoolManager::remove_pool(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex_);

        pools_.erase(
            std::remove_if(pools_.begin(), pools_.end(),
                           [&name](const PoolEntry &entry) { return entry.name == name; }),
            pools_.end()
        );
    }

    void MultiPoolManager::set_pool_priority(const std::string &name, int priority) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = std::find_if(pools_.begin(), pools_.end(),
                               [&name](const PoolEntry &entry) { return entry.name == name; });

        if (it != pools_.end()) {
            it->priority = priority;
            sort_pools_by_priority();
        }
    }

    void MultiPoolManager::enable_pool(const std::string &name, bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = std::find_if(pools_.begin(), pools_.end(),
                               [&name](const PoolEntry &entry) { return entry.name == name; });

        if (it != pools_.end()) {
            it->enabled = enable;
        }
    }

    bool MultiPoolManager::start_mining(const PoolMiningSystem::Config &mining_config) {
        if (running_.load()) {
            return true;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if (pools_.empty()) {
            std::cerr << "No pools configured" << std::endl;
            return false;
        }

        base_mining_config_ = mining_config;
        running_ = true;

        // Try to start with the first enabled pool
        if (!try_next_pool()) {
            std::cerr << "Failed to connect to any pool" << std::endl;
            running_ = false;
            return false;
        }

        // Start failover monitor
        failover_thread_ = std::thread(&MultiPoolManager::failover_monitor, this);

        return true;
    }

    void MultiPoolManager::stop_mining() {
        if (!running_.load()) {
            return;
        }

        running_ = false;

        // Stop current mining
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto &pool: pools_) {
                if (pool.mining_system) {
                    pool.mining_system->stop();
                }
            }
        }

        // Join failover thread
        if (failover_thread_.joinable()) {
            failover_thread_.join();
        }
    }

    std::string MultiPoolManager::get_active_pool() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_pool_;
    }

    std::map<std::string, PoolMiningSystem::PoolMiningStats> MultiPoolManager::get_all_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::map<std::string, PoolMiningSystem::PoolMiningStats> all_stats;

        for (const auto &pool: pools_) {
            if (pool.mining_system) {
                all_stats[pool.name] = pool.mining_system->get_stats();
            }
        }

        return all_stats;
    }

    void MultiPoolManager::failover_monitor() {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));

            std::lock_guard<std::mutex> lock(mutex_);

            // Check if current pool is still connected
            auto it = std::find_if(pools_.begin(), pools_.end(),
                                   [this](const PoolEntry &entry) { return entry.name == active_pool_; });

            if (it != pools_.end() && it->mining_system) {
                auto stats = it->mining_system->get_stats();
                if (!stats.connected || !stats.authenticated) {
                    std::cout << "Pool " << active_pool_ << " disconnected, failing over..." << std::endl;
                    try_next_pool();
                }
            }
        }
    }

    bool MultiPoolManager::try_next_pool() {
        // Stop current pool if any
        for (auto &pool: pools_) {
            if (pool.name == active_pool_ && pool.mining_system) {
                pool.mining_system->stop();
                pool.mining_system.reset();
            }
        }

        // Try each pool in priority order
        for (auto &pool: pools_) {
            if (!pool.enabled) {
                continue;
            }

            std::cout << "Trying pool: " << pool.name << std::endl;

            // Create mining config for this pool
            auto config = base_mining_config_;
            config.pool_config = pool.config;

            // Create and start mining system
            pool.mining_system = std::make_unique<PoolMiningSystem>(config);
            if (pool.mining_system->start()) {
                active_pool_ = pool.name;
                std::cout << "Connected to pool: " << pool.name << std::endl;
                return true;
            }

            // Failed, clean up
            pool.mining_system.reset();
        }

        return false;
    }

    void MultiPoolManager::sort_pools_by_priority() {
        std::sort(pools_.begin(), pools_.end(),
                  [](const PoolEntry &a, const PoolEntry &b) {
                      return a.priority < b.priority;
                  });
    }
} // namespace MiningPool
