#include "pool_integration.hpp"
#include "../logging/logger.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <deque>

namespace MiningPool {
    // Helper function to convert binary hash to hex string
    static std::string hash_to_hex(const uint32_t hash[5]) {
        std::string result;
        result.reserve(40); // 5 * 8 characters
        for (int i = 0; i < 5; i++) {
            char hex_chars[9]; // 8 chars + null terminator
            snprintf(hex_chars, sizeof(hex_chars), "%08x", hash[i]);
            result += hex_chars;
        }

        return result;
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

        LOG_INFO("POOL", "Starting pool mining system...");

        // Create pool client
        pool_client_ = std::make_unique<PoolClient>(config_.pool_config, this);

        // Initialize mining system with proper configuration
        MiningSystem::Config mining_config;

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
                    LOG_ERROR("POOL", "Failed to initialize multi-GPU manager");
                    return false;
                }

                LOG_INFO("POOL", "Initialized ", device_count, " GPUs for pool mining");
            } else {
                // Single GPU fallback
                mining_config.device_id = 0;
            }
        } else if (!config_.gpu_ids.empty()) {
            if (config_.gpu_ids.size() > 1) {
                // Multiple specific GPUs
                multi_gpu_manager_ = std::make_unique<MultiGPUManager>();
                if (!multi_gpu_manager_->initialize(config_.gpu_ids)) {
                    LOG_ERROR("POOL", "Failed to initialize multi-GPU manager");
                    return false;
                }
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
                LOG_ERROR("POOL", "Failed to initialize mining system");
                return false;
            }
        }

        // Connect to pool
        if (!pool_client_->connect()) {
            LOG_ERROR("POOL", "Failed to connect to pool");
            return false;
        }

        running_ = true;

        // Start worker threads
        mining_thread_ = std::thread(&PoolMiningSystem::mining_loop, this);
        share_scanner_thread_ = std::thread(&PoolMiningSystem::share_scanner_loop, this);
        stats_reporter_thread_ = std::thread(&PoolMiningSystem::stats_reporter_loop, this);
        share_submission_thread_ = std::thread(&PoolMiningSystem::share_submission_loop, this);

        return true;
    }

    void PoolMiningSystem::stop() {
        if (!running_.load()) {
            return;
        }

        LOG_INFO("POOL", "Stopping pool mining system...");

        running_ = false;
        mining_active_ = false;

        // Notify condition variables
        job_cv_.notify_all();
        share_cv_.notify_all();
        results_cv_.notify_all();

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
        if (share_submission_thread_.joinable()) {
            share_submission_thread_.join();
        }

        LOG_INFO("POOL", "Pool mining system stopped");
    }

    void PoolMiningSystem::mining_loop() {
    LOG_INFO("MINING", "Mining loop started");

    while (running_.load()) {
        std::optional<PoolJob> current_pool_job;
        std::optional<MiningJob> current_mining_job_local;
        std::string current_job_id;
        uint64_t current_generation;

        // Wait for a job
        {
            std::unique_lock<std::mutex> lock(job_mutex_);
            job_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return current_job_.has_value() || !running_.load();
            });

            if (!running_.load()) break;
            if (!current_job_.has_value()) {
                LOG_DEBUG("MINING", "No job available, waiting...");
                continue;
            }

            current_pool_job = current_job_;
            current_mining_job_local = current_mining_job_;
            current_job_id = current_job_->job_id;
            current_generation = job_generation_.load();
        }

        if (!current_mining_job_local.has_value()) {
            LOG_WARN("MINING", "Job available but mining job conversion failed");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Reset the restart flag
        should_restart_mining_ = false;

        // Store this mining batch
        {
            std::lock_guard<std::mutex> lock(batch_mutex_);
            MiningBatch batch;
            batch.job_id = current_job_id;
            batch.mining_job = *current_mining_job_local;
            batch.start_time = std::chrono::steady_clock::now();

            active_batches_.push_back(batch);

            while (active_batches_.size() > MAX_ACTIVE_BATCHES) {
                active_batches_.pop_front();
            }
        }

        // Set up result callback
        auto job_id_for_callback = current_job_id;
        auto result_callback = [this, job_id_for_callback](const std::vector<MiningResult> &results) {
            process_mining_results(results, job_id_for_callback);
        };

        if (mining_system_) {
            mining_system_->setResultCallback(result_callback);
            mining_system_->resetState();  // Reset counters for new job
        } else if (multi_gpu_manager_) {
            multi_gpu_manager_->setResultCallback(result_callback);
        }

        LOG_INFO("MINING", "Starting mining with job: ", current_job_id);
        LOG_INFO("MINING", "Target difficulty: ", current_mining_job_local->difficulty);

        // Log the target pattern
        std::stringstream target_ss;
        for (int i = 0; i < 5; i++) {
            target_ss << std::hex << std::setw(8) << std::setfill('0')
                     << current_mining_job_local->target_hash[i];
            if (i < 4) target_ss << " ";
        }
        LOG_INFO("MINING", "Mining with target pattern: ", target_ss.str());

        try {
            // Run mining in small chunks to allow checking for updates
            const int chunk_seconds = 1; // Check every second
            const int max_chunks = 300;  // Maximum 5 minutes total

            for (int chunk = 0; chunk < max_chunks; chunk++) {
                // Check if we should stop (new job arrived)
                {
                    std::lock_guard<std::mutex> lock(job_mutex_);
                    if (job_generation_.load() != current_generation) {
                        LOG_INFO("MINING", "New job detected, switching...");
                        break;
                    }
                }

                // Also check the restart flag
                if (should_restart_mining_.load()) {
                    LOG_INFO("MINING", "Mining restart requested");
                    break;
                }

                // Run mining for this chunk
                if (mining_system_) {
                    mining_system_->runMiningLoop(*current_mining_job_local, chunk_seconds);
                } else if (multi_gpu_manager_) {
                    multi_gpu_manager_->runMining(*current_mining_job_local, chunk_seconds);
                }

                // If mining stopped early (due to stopMining()), break
                if (mining_system_ && !mining_system_->shouldContinueMining()) {
                    LOG_INFO("MINING", "Mining stopped early, checking for new job");
                    break;
                }
            }

        } catch (const std::exception &e) {
            LOG_ERROR("MINING", "Mining error: ", e.what());
        }

        LOG_INFO("MINING", "Mining batch completed for job: ", current_job_id);
    }

    LOG_INFO("MINING", "Mining loop stopped");
}

    void PoolMiningSystem::share_scanner_loop() {
        LOG_INFO("SCANNER", "Share scanner loop started");

        while (running_.load()) {
            {
                std::unique_lock<std::mutex> lock(results_mutex_);
                results_cv_.wait_for(lock, std::chrono::milliseconds(config_.share_scan_interval_ms), [this] {
                    return !current_mining_results_with_job_.empty() || !running_.load();
                });
            }

            if (!running_.load()) break;

            try {
                scan_for_shares();
            } catch (const std::exception &e) {
                LOG_ERROR("SCANNER", "Share scanning exception: ", e.what());
            }

            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        LOG_INFO("SCANNER", "Share scanner loop stopped");
    }

    void PoolMiningSystem::share_submission_loop() {
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(share_mutex_);
            share_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !share_queue_.empty() || !running_.load();
            });

            if (!running_.load()) break;

            while (!share_queue_.empty()) {
                Share share = share_queue_.front();
                share_queue_.pop();
                lock.unlock();

                // Send via pool client
                if (pool_client_ && pool_client_->is_connected()) {
                    pool_client_->submit_share(share);

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
        std::vector<MiningResultWithJob> results_to_scan; {
            std::lock_guard<std::mutex> lock(results_mutex_);
            if (current_mining_results_with_job_.empty()) return;
            results_to_scan = std::move(current_mining_results_with_job_);
            current_mining_results_with_job_.clear();
        }

        LOG_DEBUG("SHARE", "Scanning ", results_to_scan.size(), " results for shares");

        // Group results by job ID
        std::unordered_map<std::string, std::vector<MiningResult> > results_by_job;
        for (const auto &item: results_to_scan) {
            results_by_job[item.job_id].push_back(item.result);
        }

        // Process each job's results
        for (const auto &[job_id, results]: results_by_job) {
            // Find the job data for this job ID
            std::optional<PoolJob> job_for_validation; {
                std::lock_guard<std::mutex> lock(job_mutex_);
                auto it = active_jobs_.find(job_id);
                if (it != active_jobs_.end()) {
                    job_for_validation = it->second;
                }
            }

            if (!job_for_validation.has_value()) {
                LOG_WARN("SHARE", "No job data found for job ID: ", job_id, ", discarding ", results.size(),
                         " results");
                continue;
            }

            uint32_t job_difficulty = job_for_validation->job_data.target_difficulty;
            uint32_t min_difficulty_for_shares = std::min(job_difficulty, config_.min_share_difficulty);

            LOG_DEBUG("SHARE", "Processing ", results.size(), " results for job ", job_id);
            LOG_DEBUG("SHARE", "Job difficulty: ", job_difficulty, ", min share difficulty: ",
                      min_difficulty_for_shares);

            for (const auto &result: results) {
                LOG_DEBUG("SHARE", "Checking result: nonce=0x", std::hex, result.nonce,
                          std::dec, ", bits=", result.matching_bits);

                if (result.matching_bits >= min_difficulty_for_shares) {
                    LOG_INFO("SHARE", "Found valid share: ", result.matching_bits, " bits (min: ",
                             min_difficulty_for_shares, ") for job ", job_id);

                    // Create share with the correct job ID
                    Share share;
                    share.job_id = job_id; // Use the job ID this result was found for
                    share.nonce = result.nonce;
                    share.hash = hash_to_hex(result.hash);
                    share.matching_bits = result.matching_bits;
                    share.found_time = std::chrono::steady_clock::now();

                    submit_share_direct(share);
                } else {
                    LOG_DEBUG("SHARE", "Result ", result.matching_bits, " bits below threshold ",
                              min_difficulty_for_shares);
                }
            }
        }
    }

    void PoolMiningSystem::process_mining_results(const std::vector<MiningResult> &results, const std::string &job_id) {
        if (results.empty()) return;

        LOG_DEBUG("PROCESS", "Processing ", results.size(), " results for job ", job_id);

        // Store results with their job ID
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            for (const auto &result: results) {
                MiningResultWithJob result_with_job;
                result_with_job.result = result;
                result_with_job.job_id = job_id;
                current_mining_results_with_job_.push_back(result_with_job);
            }
            LOG_DEBUG("PROCESS", "Stored results, total pending: ", current_mining_results_with_job_.size());
        }

        // Notify share scanner
        results_cv_.notify_one();
    }

    void PoolMiningSystem::submit_share_direct(const Share &share) {
        LOG_INFO("SHARE", "Submitting share: nonce=0x", std::hex, share.nonce, std::dec,
                 ", bits=", share.matching_bits);

        // Verify job still exists
        {
            std::lock_guard<std::mutex> lock(job_mutex_);
            if (active_jobs_.find(share.job_id) == active_jobs_.end()) {
                LOG_ERROR("SHARE", "Job ", share.job_id, " no longer active, discarding share");
                return;
            }
        }

        LOG_INFO("SHARE", "Formatted share data:");
        LOG_INFO("SHARE", "  Job ID: ", share.job_id);
        LOG_INFO("SHARE", "  Nonce: 0x", std::hex, share.nonce, std::dec);
        LOG_INFO("SHARE", "  Hash: ", share.hash);
        LOG_INFO("SHARE", "  Bits: ", share.matching_bits);

        // Submit directly via pool client
        if (pool_client_ && pool_client_->is_connected() && pool_client_->is_authenticated()) {
            try {
                pool_client_->submit_share(share);

                // Update stats
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.shares_submitted++;

                LOG_INFO("SHARE", "Share submitted successfully");
            } catch (const std::exception &e) {
                LOG_ERROR("SHARE", "Exception during share submission: ", e.what());
            }
        } else {
            LOG_ERROR("SHARE", "Cannot submit share - pool client not ready");
        }
    }

    MiningJob PoolMiningSystem::convert_to_mining_job(const JobMessage &job_msg) {
        MiningJob mining_job{};

        // The prefix_data now contains the unique salted preimage for this worker
        auto prefix_bytes = MiningPool::Utils::hex_to_bytes(job_msg.prefix_data);
        auto target_bytes = MiningPool::Utils::hex_to_bytes(job_msg.target_pattern);

        if (target_bytes.size() != 20) {
            LOG_ERROR("POOL", "Invalid target pattern size: ", target_bytes.size());
            return mining_job;
        }

        // Clear the base message
        std::memset(mining_job.base_message, 0, 32);

        // Copy the salted preimage to base message
        if (!prefix_bytes.empty()) {
            size_t copy_size = std::min(prefix_bytes.size(), size_t(32));
            std::memcpy(mining_job.base_message, prefix_bytes.data(), copy_size);
            LOG_DEBUG("POOL", "Using salted preimage of ", copy_size, " bytes");
        }

        // Handle suffix data if present (usually empty for salted preimages)
        if (!job_msg.suffix_data.empty()) {
            auto suffix_bytes = MiningPool::Utils::hex_to_bytes(job_msg.suffix_data);
            if (!suffix_bytes.empty() && prefix_bytes.size() + suffix_bytes.size() <= 32) {
                size_t suffix_offset = prefix_bytes.size();
                std::memcpy(mining_job.base_message + suffix_offset, suffix_bytes.data(), suffix_bytes.size());
                LOG_DEBUG("POOL", "Added ", suffix_bytes.size(), " suffix bytes at offset ", suffix_offset);
            }
        }

        // Convert target hash to uint32_t array
        for (int i = 0; i < 5; i++) {
            mining_job.target_hash[i] = (static_cast<uint32_t>(target_bytes[i * 4]) << 24) |
                                        (static_cast<uint32_t>(target_bytes[i * 4 + 1]) << 16) |
                                        (static_cast<uint32_t>(target_bytes[i * 4 + 2]) << 8) |
                                        static_cast<uint32_t>(target_bytes[i * 4 + 3]);
        }

        mining_job.difficulty = job_msg.target_difficulty;
        mining_job.nonce_offset = job_msg.nonce_start;

        // Log epoch information if available
        if (job_msg.extra_data.contains("epoch_number")) {
            LOG_INFO("POOL", "Mining for epoch #", job_msg.extra_data["epoch_number"].get<int>());
        }
        if (job_msg.extra_data.contains("epoch_target_hash")) {
            LOG_DEBUG("POOL", "Epoch target: ", job_msg.extra_data["epoch_target_hash"].get<std::string>());
        }
        if (job_msg.extra_data.contains("worker_salt")) {
            LOG_DEBUG("POOL", "Worker salt: ", job_msg.extra_data["worker_salt"].get<std::string>().substr(0, 16),
                      "...");
        }

        // Verify target pattern
        std::string target_hex_verify;
        for (int i = 0; i < 5; i++) {
            char buf[9];
            snprintf(buf, sizeof(buf), "%08x", mining_job.target_hash[i]);
            target_hex_verify += buf;
        }
        LOG_DEBUG("POOL", "Target pattern set to: ", target_hex_verify);

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
        std::lock_guard<std::mutex> lock(stats_mutex_);

        if (mining_system_) {
            auto mining_stats = mining_system_->getStats();
            stats_.hashrate = mining_stats.hash_rate;
            stats_.total_hashes = mining_stats.hashes_computed;
        }

        // Update current difficulty from the actual job
        stats_.current_difficulty = current_difficulty_.load();

        // Calculate uptime
        auto now = std::chrono::steady_clock::now();
        stats_.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        // Calculate share success rate
        if (stats_.shares_submitted > 0) {
            stats_.share_success_rate = static_cast<double>(stats_.shares_accepted) / stats_.shares_submitted;
        } else {
            stats_.share_success_rate = 0.0;
        }
    }

    void PoolMiningSystem::handle_reconnect() {
        LOG_INFO("POOL", "Handling reconnection...");

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
            LOG_INFO("POOL", "Requesting difficulty increase to ", new_diff,
                     " (current avg share time: ", avg_share_time, "s)");

            // Some pools support client-requested difficulty adjustments
            // This would need protocol support
        } else if (avg_share_time > config_.target_share_time * 2.0) {
            // Too few shares, request lower difficulty
            uint32_t new_diff = std::max(config_.min_share_difficulty, current_difficulty_.load() - 1);
            LOG_INFO("POOL", "Requesting difficulty decrease to ", new_diff,
                     " (current avg share time: ", avg_share_time, "s)");
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
        LOG_INFO("POOL", Color::GREEN, "Connected to mining pool", Color::RESET);

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.connected = true;
    }

    void PoolMiningSystem::on_disconnected(const std::string &reason) {
        LOG_WARN("POOL", Color::RED, "Disconnected from pool: ", reason, Color::RESET);

        mining_active_ = false;
        job_cv_.notify_all();

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.connected = false;
        stats_.authenticated = false;
    }

    void PoolMiningSystem::on_error(ErrorCode code, const std::string &message) {
        LOG_ERROR("POOL", "Pool error (", static_cast<int>(code), "): ", message);
    }

    void PoolMiningSystem::on_authenticated(const std::string &worker_id) {
        LOG_INFO("POOL", Color::GREEN, "Authenticated as worker: ", worker_id, Color::RESET);

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.authenticated = true;
        stats_.worker_id = worker_id;
    }

    void PoolMiningSystem::on_auth_failed(ErrorCode code, const std::string &reason) {
        LOG_ERROR("POOL", Color::RED, "Authentication failed: ", reason, Color::RESET);

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.authenticated = false;

        // Stop mining on auth failure
        mining_active_ = false;
    }

    void PoolMiningSystem::on_new_job(const PoolJob &job) {
    LOG_INFO("POOL", "New job received: ", job.job_id, " (difficulty: ", job.job_data.target_difficulty, ")");

    bool should_interrupt = false;

    // Store job in active jobs map
    {
        std::lock_guard<std::mutex> lock(job_mutex_);

        // Check if we need to interrupt current mining
        if (job.job_data.clean_jobs && current_job_.has_value()) {
            LOG_WARN("POOL", "Clean jobs requested - interrupting current mining");
            should_interrupt = true;
        }

        // Clear jobs if requested
        if (job.job_data.clean_jobs) {
            active_jobs_.clear();

            // Clear pending results
            {
                std::lock_guard<std::mutex> results_lock(results_mutex_);
                current_mining_results_with_job_.clear();
            }
        }

        // Add the new job
        active_jobs_[job.job_id] = job;

        // Update current job
        current_job_ = job;
        current_mining_job_ = convert_to_mining_job(job.job_data);
        current_job_id_for_mining_ = job.job_id;

        // Increment job generation
        job_generation_++;

        // Clean up old jobs if needed
        while (!job.job_data.clean_jobs && active_jobs_.size() > 10) {
            std::string oldest_id;
            auto oldest_time = std::chrono::steady_clock::now();

            for (const auto &[id, j] : active_jobs_) {
                if (j.received_time < oldest_time) {
                    oldest_time = j.received_time;
                    oldest_id = id;
                }
            }

            if (!oldest_id.empty()) {
                active_jobs_.erase(oldest_id);
            }
        }
    }

    // Update difficulty
    current_difficulty_.store(job.job_data.target_difficulty);

    // Signal mining restart if needed
    if (should_interrupt) {
        should_restart_mining_ = true;

        // Interrupt the current mining
        if (mining_system_) {
            mining_system_->stopMining();
        } else if (multi_gpu_manager_) {
            multi_gpu_manager_->stopMining();
        }
    }

    // Notify mining thread
    job_cv_.notify_all();
}

    void PoolMiningSystem::on_job_cancelled(const std::string &job_id) {
        LOG_INFO("POOL", "Job cancelled: ", job_id);

        std::lock_guard<std::mutex> lock(job_mutex_);

        // Remove from active jobs
        active_jobs_.erase(job_id);

        // If this was the current job, stop mining immediately
        if (current_job_.has_value() && current_job_->job_id == job_id) {
            LOG_WARN("POOL", "Current job was cancelled, stopping mining");
            current_job_.reset();
            current_mining_job_.reset();
            mining_active_ = false;

            // Clear any pending results for this job
            {
                std::lock_guard<std::mutex> results_lock(results_mutex_);
                current_mining_results_with_job_.erase(
                    std::ranges::remove_if(current_mining_results_with_job_,
                                           [&job_id](const MiningResultWithJob& r) {
                                               return r.job_id == job_id;
                                           }).begin(),
                    current_mining_results_with_job_.end()
                );
            }

            job_cv_.notify_all();
        }
    }

    void PoolMiningSystem::on_share_accepted(const ShareResultMessage &result) {
        LOG_INFO("POOL", Color::BRIGHT_GREEN, "Share accepted! Difficulty: ",
                 result.difficulty_credited, Color::RESET);

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.shares_accepted++;

            // Track share times for vardiff
            share_times_.push_back(std::chrono::steady_clock::now());
            if (share_times_.size() > 100) {
                share_times_.pop_front();
            }

            if (!result.message.empty()) {
                LOG_INFO("POOL", Color::BRIGHT_YELLOW, "Pool message: ", result.message, Color::RESET);

                // Check for special messages
                if (result.message.find("High-value contribution") != std::string::npos) {
                    LOG_INFO("POOL", Color::BRIGHT_CYAN,
                             "*** HIGH-VALUE EPOCH CONTRIBUTION FOUND! ***", Color::RESET);
                }
            }

            // Log share value if provided
            if (result.share_value > 0) {
                LOG_DEBUG("POOL", "Share value: ", result.share_value);
            }
        }
    }

    void PoolMiningSystem::on_share_rejected(const ShareResultMessage &result) {
        LOG_WARN("POOL", Color::RED, "Share rejected: ", result.message, Color::RESET);

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.shares_rejected++;
        }
    }

    void PoolMiningSystem::on_difficulty_changed(uint32_t new_difficulty) {
        LOG_INFO("POOL", Color::BRIGHT_MAGENTA, "Difficulty adjusted to: ", new_difficulty, Color::RESET);

        current_difficulty_ = new_difficulty;

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.current_difficulty = new_difficulty;
    }

    void PoolMiningSystem::on_pool_status(const PoolStatusMessage &status) {
        LOG_INFO("POOL", "Pool status - Workers: ", status.connected_workers,
                 ", Hashrate: ", status.total_hashrate / 1e9, " GH/s",
                 ", Epoch: #", status.current_epoch_number,
                 ", Epoch shares: ", status.current_epoch_shares);

        // Display epoch progress if available
        if (status.extra_info.contains("epoch_info") && !status.extra_info["epoch_info"].is_null()) {
            auto epoch_info = status.extra_info["epoch_info"];
            if (epoch_info.contains("current_epoch_target_hash")) {
                LOG_DEBUG("POOL", "Current epoch target: ", epoch_info["current_epoch_target_hash"].get<std::string>());
            }
            if (epoch_info.contains("blocks_per_epoch")) {
                LOG_DEBUG("POOL", "Blocks per epoch: ", epoch_info["blocks_per_epoch"].get<int>());
            }
            if (epoch_info.contains("epochs_until_payout")) {
                LOG_INFO("POOL", "Epochs until payout: ", epoch_info["epochs_until_payout"].get<int>());
            }
        }

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
            LOG_ERROR("MULTI_POOL", "No pools configured");
            return false;
        }

        base_mining_config_ = mining_config;
        running_ = true;

        // Try to start with the first enabled pool
        if (!try_next_pool()) {
            LOG_ERROR("MULTI_POOL", "Failed to connect to any pool");
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
                    LOG_WARN("MULTI_POOL", "Pool ", active_pool_, " disconnected, failing over...");
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

            LOG_INFO("MULTI_POOL", "Trying pool: ", pool.name);

            // Create mining config for this pool
            auto config = base_mining_config_;
            config.pool_config = pool.config;

            // Create and start mining system
            pool.mining_system = std::make_unique<PoolMiningSystem>(config);
            if (pool.mining_system->start()) {
                active_pool_ = pool.name;
                LOG_INFO("MULTI_POOL", Color::GREEN, "Connected to pool: ", pool.name, Color::RESET);
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

