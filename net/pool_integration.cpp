#include "pool_integration.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <deque>

namespace MiningPool {
    // Helper function to convert binary hash to hex string
    static std::string hash_to_hex(const uint32_t hash[5]) {
        std::stringstream ss;
        for (int i = 0; i < 5; i++) {
            ss << std::hex << std::setfill('0') << std::setw(8) << hash[i];
        }
        return ss.str();
    }

    // Helper function to convert hex string to binary
    static void hex_to_hash(const std::string &hex, uint32_t hash[5]) {
        for (int i = 0; i < 5 && i * 8 < hex.length(); i++) {
            hash[i] = std::stoul(hex.substr(i * 8, 8), nullptr, 16);
        }
    }

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

        // Initialize mining system with proper configuration
        MiningSystem::Config mining_config;

        // Create result callback for share submission
        auto result_callback = [this](const std::vector<MiningResult> &results) {
            process_mining_results(results);
        };

        // Configure GPU setup
        if (config_.use_all_gpus) {
            // For multi-GPU, we'll use a MultiGPUManager
            int device_count;
            gpuGetDeviceCount(&device_count);

            if (device_count > 1) {
                // Initialize multi-GPU manager
                multi_gpu_manager_ = std::make_unique<MultiGPUManager>();
                std::vector<int> gpu_ids;
                for (int i = 0; i < device_count; i++) {
                    gpu_ids.push_back(i);
                }

                if (!multi_gpu_manager_->initialize(gpu_ids)) {
                    std::cerr << "Failed to initialize multi-GPU manager" << std::endl;
                    return false;
                }

                // Set the result callback
                multi_gpu_manager_->setResultCallback(result_callback);

                std::cout << "Initialized " << device_count << " GPUs for pool mining" << std::endl;
            } else {
                // Single GPU fallback
                mining_config.device_id = 0;
            }
        } else if (!config_.gpu_ids.empty()) {
            if (config_.gpu_ids.size() > 1) {
                // Multiple specific GPUs
                multi_gpu_manager_ = std::make_unique<MultiGPUManager>();
                if (!multi_gpu_manager_->initialize(config_.gpu_ids)) {
                    std::cerr << "Failed to initialize multi-GPU manager" << std::endl;
                    return false;
                }
                multi_gpu_manager_->setResultCallback(result_callback);
            } else {
                // Single specific GPU
                mining_config.device_id = config_.gpu_ids[0];
            }
        } else {
            // Single GPU specified by ID
            mining_config.device_id = config_.gpu_id;
        }

        // Initialize single GPU mining system if not using multi-GPU
        if (!multi_gpu_manager_) {
            mining_system_ = std::make_unique<MiningSystem>(mining_config);
            if (!mining_system_->initialize()) {
                std::cerr << "Failed to initialize mining system" << std::endl;
                return false;
            }
            mining_system_->setResultCallback(result_callback);
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

        // Notify condition variables
        job_cv_.notify_all();
        share_cv_.notify_all();

        // Disconnect from pool
        if (pool_client_) {
            pool_client_->disconnect();
        }

        // Stop mining
        if (mining_system_ || multi_gpu_manager_) {
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
            // Wait for an active job
            std::unique_lock<std::mutex> lock(job_mutex_);
            job_cv_.wait(lock, [this] {
                return !running_.load() || (current_job_.has_value() && mining_active_.load());
            });

            if (!running_.load()) break;

            if (!current_mining_job_.has_value()) {
                continue;
            }

            auto job = current_mining_job_.value();
            lock.unlock();

            // Run mining based on system type
            if (multi_gpu_manager_) {
                // Multi-GPU mining with real-time result callbacks
                const uint32_t batch_duration = 30; // 30 seconds per batch

                // Store current job ID for share submission
                current_job_id_for_mining_ = current_job_->job_id;

                // Run mining - results will be processed via callback
                multi_gpu_manager_->runMining(job, batch_duration);
            } else if (mining_system_) {
                // Single GPU mining with real-time result callbacks
                const uint32_t batch_duration = 30;

                // Store current job ID for share submission
                current_job_id_for_mining_ = current_job_->job_id;

                // Run mining - results will be processed via callback
                mining_system_->runMiningLoop(job, batch_duration);
            }
        }
    }

    void PoolMiningSystem::share_scanner_loop() {
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(share_mutex_);
            share_cv_.wait_for(lock, std::chrono::milliseconds(config_.share_scan_interval_ms), [this] {
                return !running_.load() || !share_queue_.empty();
            });

            if (!running_.load()) break;

            // Process all queued shares
            while (!share_queue_.empty()) {
                auto share = share_queue_.front();
                share_queue_.pop();
                lock.unlock();

                // Submit to pool
                pool_client_->submit_share(share);

                // Track submission
                {
                    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                    stats_.shares_submitted++;
                }

                lock.lock();
            }
        }
    }

    void PoolMiningSystem::stats_reporter_loop() {
        auto last_hashrate_report = std::chrono::steady_clock::now();
        auto last_difficulty_check = std::chrono::steady_clock::now();

        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            if (!pool_client_->is_connected() || !mining_active_.load()) {
                continue;
            }

            auto now = std::chrono::steady_clock::now();

            // Report hashrate periodically
            if (now - last_hashrate_report >= std::chrono::seconds(30)) {
                last_hashrate_report = now;

                HashrateReportMessage report;

                // Get mining stats
                if (multi_gpu_manager_) {
                    // Get stats from multi-GPU - need to calculate from elapsed time
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
                    if (elapsed.count() > 0) {
                        report.hashrate = stats_.hashrate; // Updated by update_stats()
                    }
                    report.gpu_count = config_.gpu_ids.empty() ? 1 : config_.gpu_ids.size();
                } else if (mining_system_) {
                    auto mining_stats = mining_system_->getStats();
                    report.hashrate = mining_stats.hash_rate;
                    report.gpu_count = 1;
                }

                report.shares_submitted = stats_.shares_submitted;
                report.shares_accepted = stats_.shares_accepted;
                report.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                    now - start_time_
                ).count();

                // Add GPU stats
                nlohmann::json gpu_stats;
                if (multi_gpu_manager_) {
                    for (uint32_t i = 0; i < report.gpu_count; i++) {
                        gpu_stats["gpu_" + std::to_string(i)]["hashrate"] = report.hashrate / report.gpu_count;
                        gpu_stats["gpu_" + std::to_string(i)]["temperature"] = 0; // TODO: Add temp monitoring
                    }
                } else {
                    gpu_stats["gpu_0"]["hashrate"] = report.hashrate;
                    gpu_stats["gpu_0"]["temperature"] = 0; // TODO: Add temp monitoring
                }
                report.gpu_stats = gpu_stats;

                pool_client_->report_hashrate(report);
            }

            // Check for vardiff adjustment
            if (config_.enable_vardiff && now - last_difficulty_check >= std::chrono::seconds(60)) {
                last_difficulty_check = now;
                adjust_local_difficulty();
            }

            // Update internal stats
            update_stats();
        }
    }

    void PoolMiningSystem::scan_for_shares() {
        // Not needed with callback-based approach
        // Shares are processed in real-time via process_mining_results
    }

    void PoolMiningSystem::processAccumulatedResults() {
        // Not needed with callback-based approach
    }

    void PoolMiningSystem::process_mining_results(const std::vector<MiningResult> &results) {
        uint32_t pool_difficulty = current_difficulty_.load();

        for (const auto &result: results) {
            if (result.matching_bits >= pool_difficulty) {
                submit_share(result);
            }
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
        share.hash = hash_to_hex(result.hash);
        share.matching_bits = result.matching_bits;
        share.found_time = std::chrono::steady_clock::now();

        // Queue share for submission
        {
            std::lock_guard<std::mutex> share_lock(share_mutex_);
            share_queue_.push(share);
            share_cv_.notify_one();
        }

        // Track locally
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);

            // Track share timing for vardiff
            share_times_.push_back(share.found_time);

            // Keep only recent share times (last 100 shares)
            if (share_times_.size() > 100) {
                share_times_.pop_front();
            }
        }
    }

    MiningJob PoolMiningSystem::convert_to_mining_job(const JobMessage &job_msg) {
        MiningJob mining_job;

        // Convert prefix data (hex string) to binary
        auto prefix_bytes = Utils::hex_to_bytes(job_msg.prefix_data);
        if (prefix_bytes.size() >= 32) {
            std::copy(prefix_bytes.begin(), prefix_bytes.begin() + 32, mining_job.base_message);
        }

        // Convert target pattern (hex string) to binary hash
        hex_to_hash(job_msg.target_pattern, mining_job.target_hash);

        mining_job.difficulty = job_msg.target_difficulty;
        mining_job.nonce_offset = job_msg.nonce_start;

        return mining_job;
    }

    void PoolMiningSystem::update_mining_job(const PoolJob &pool_job) {
        std::lock_guard<std::mutex> lock(job_mutex_);

        current_job_ = pool_job;
        current_mining_job_ = convert_to_mining_job(pool_job.job_data);

        // Update difficulty
        current_difficulty_ = pool_job.job_data.target_difficulty;

        // Start mining if not already active
        mining_active_ = true;
        job_cv_.notify_all();
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

    void PoolMiningSystem::cleanup_mining_system() {
        if (mining_system_) {
            mining_system_.reset();
        }
        if (multi_gpu_manager_) {
            multi_gpu_manager_.reset();
        }
    }

    void PoolMiningSystem::update_stats() {
        double current_hashrate = 0.0;
        uint64_t total_hashes = 0;

        if (multi_gpu_manager_) {
            // Estimate based on share submission rate and difficulty
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_
            );
            if (elapsed.count() > 0 && stats_.shares_submitted > 0) {
                // Better estimation: use share rate and difficulty
                double shares_per_second = static_cast<double>(stats_.shares_submitted) / elapsed.count();
                current_hashrate = shares_per_second * std::pow(2.0, static_cast<double>(current_difficulty_.load()));
            }
        } else if (mining_system_) {
            auto mining_stats = mining_system_->getStats();
            current_hashrate = mining_stats.hash_rate;
            total_hashes = mining_stats.hashes_computed;
        }

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.hashrate = current_hashrate;
        stats_.total_hashes = total_hashes;
    }

    void PoolMiningSystem::handle_reconnect() {
        std::cout << "Handling reconnection..." << std::endl;

        // Stop mining temporarily
        mining_active_ = false;

        // Clear current job
        {
            std::lock_guard<std::mutex> lock(job_mutex_);
            current_job_.reset();
            current_mining_job_.reset();
        }

        // Clear pending shares
        {
            std::lock_guard<std::mutex> lock(share_mutex_);
            std::queue<Share> empty;
            std::swap(share_queue_, empty);
        }

        // Pool client will handle the actual reconnection
    }

    void PoolMiningSystem::adjust_local_difficulty() {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        if (share_times_.size() < 10) {
            // Not enough data to adjust
            return;
        }

        // Calculate average time between shares
        double total_time = 0;
        for (size_t i = 1; i < share_times_.size(); i++) {
            auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                share_times_[i] - share_times_[i - 1]
            ).count();
            total_time += time_diff;
        }

        double avg_share_time = total_time / (share_times_.size() - 1);

        // Adjust difficulty to target share time
        if (avg_share_time < config_.target_share_time * 0.5) {
            // Too many shares, request higher difficulty
            uint32_t new_diff = current_difficulty_.load() + 1;
            std::cout << "Requesting difficulty increase to " << new_diff
                    << " (current avg share time: " << avg_share_time << "s)" << std::endl;

            // Some pools support client-requested difficulty adjustments
            // This would need protocol support
        } else if (avg_share_time > config_.target_share_time * 2.0) {
            // Too few shares, request lower difficulty
            uint32_t new_diff = std::max(config_.min_share_difficulty, current_difficulty_.load() - 1);
            std::cout << "Requesting difficulty decrease to " << new_diff
                    << " (current avg share time: " << avg_share_time << "s)" << std::endl;
        }
    }

    uint32_t PoolMiningSystem::calculate_optimal_scan_difficulty() {
        // Based on hashrate and target share time
        if (stats_.hashrate > 0) {
            // Calculate expected time to find a share at various difficulties
            double hashes_for_target_time = stats_.hashrate * config_.target_share_time;

            // Find difficulty that gives us approximately target_share_time
            uint32_t optimal_diff = config_.min_share_difficulty;

            while (optimal_diff < 60) {
                double expected_hashes = std::pow(2.0, static_cast<double>(optimal_diff));
                if (expected_hashes > hashes_for_target_time) {
                    break;
                }
                optimal_diff++;
            }

            return optimal_diff;
        }

        return current_difficulty_.load();
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
        job_cv_.notify_all();

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

        // Stop mining on auth failure
        mining_active_ = false;
    }

    void PoolMiningSystem::on_new_job(const PoolJob &job) {
        std::cout << "New job received: " << job.job_id
                << " (difficulty: " << job.job_data.target_difficulty << ")" << std::endl;

        update_mining_job(job);
    }

    void PoolMiningSystem::on_job_cancelled(const std::string &job_id) {
        std::cout << "Job cancelled: " << job_id << std::endl;

        std::lock_guard<std::mutex> lock(job_mutex_);
        if (current_job_.has_value() && current_job_->job_id == job_id) {
            current_job_.reset();
            current_mining_job_.reset();
            mining_active_ = false;
            job_cv_.notify_all();
        }
    }

    void PoolMiningSystem::on_share_accepted(const ShareResultMessage &result) {
        std::cout << "Share accepted! Difficulty: " << result.difficulty_credited << std::endl;

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.shares_accepted++;

            if (!result.message.empty()) {
                std::cout << "Pool message: " << result.message << std::endl;
            }
        }
    }

    void PoolMiningSystem::on_share_rejected(const ShareResultMessage &result) {
        std::cout << "Share rejected: " << result.message << std::endl;

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
        std::cout << "Pool status - Workers: " << status.connected_workers
                << ", Hashrate: " << status.total_hashrate / 1e9 << " GH/s"
                << ", Round shares: " << status.current_round_shares << std::endl;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        // Pool name might be in extra_info JSON
        if (status.extra_info.contains("pool_name")) {
            stats_.pool_name = status.extra_info["pool_name"];
        }
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
