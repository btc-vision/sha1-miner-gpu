#include "pool_integration.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <iomanip>
#include <sstream>
#include <utility>

#include "../logging/logger.hpp"
#include "config_utils.hpp"

namespace MiningPool {
    // Helper function to convert binary hash to hex string
    static auto hash_to_hex(const uint32_t hash[5]) -> std::string
    {
        std::string result;
        result.reserve(40);  // 5 * 8 characters

        for (int i = 0; i < 5; i++) {
            char hex_chars[9];  // 8 chars + null terminator
            snprintf(hex_chars, sizeof(hex_chars), "%08x", hash[i]);
            result += hex_chars;
        }

        return result;
    }

    PoolMiningSystem::PoolMiningSystem(Config config) : config_(std::move(config))
    {
        // Initialize statistics
        stats_                    = {};
        stats_.current_difficulty = config_.min_share_difficulty;
        stats_.current_bits       = config_.min_share_difficulty;
        stats_.best_share_bits    = 0;
        start_time_               = std::chrono::steady_clock::now();

        // IMPORTANT: Start job version at 0
        job_version_ = 0;
    }

    PoolMiningSystem::~PoolMiningSystem()
    {
        // Ensure proper cleanup order
        running_       = false;
        mining_active_ = false;

        // Wake up all waiting threads
        job_cv_.notify_all();
        share_cv_.notify_all();
        results_cv_.notify_all();

        // Stop pool client first
        if (pool_client_) {
            pool_client_->disconnect();
        }

        // Then stop mining
        stop();

        // Clean up any pending job updates
        const JobUpdate *update = pending_job_update_.exchange(nullptr);
        delete update;
    }

    bool PoolMiningSystem::start()
    {
        if (running_.load()) {
            return true;
        }

        LOG_INFO("POOL", "Starting pool mining system...");

        // Create pool client
        pool_client_ = std::make_unique<PoolClient>(config_.pool_config, this);

        // Initialize mining system with proper configuration
        MiningSystem::Config mining_config;

        // Create result callback for share submission
        auto result_callback = [this](const std::vector<MiningResult> &results) { process_mining_results(results); };

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

                // Pass pool's mining config if available
                if (config_.mining_config) {
                    multi_gpu_manager_->setUserConfig(config_.mining_config);
                }

                if (!multi_gpu_manager_->initialize(gpu_ids)) {
                    LOG_ERROR("POOL", "Failed to initialize multi-GPU manager");
                    return false;
                }

                // Set the result callback
                multi_gpu_manager_->setResultCallback(result_callback);

                LOG_INFO("POOL", "Initialized ", device_count, " GPUs for pool mining");
            } else {
                // Single GPU fallback
                mining_config.device_id = 0;
            }
        } else if (!config_.gpu_ids.empty()) {
            if (config_.gpu_ids.size() > 1) {
                // Multiple specific GPUs
                multi_gpu_manager_ = std::make_unique<MultiGPUManager>();
                // Pass pool's mining config if available
                if (config_.mining_config) {
                    multi_gpu_manager_->setUserConfig(config_.mining_config);
                }
                if (!multi_gpu_manager_->initialize(config_.gpu_ids)) {
                    LOG_ERROR("POOL", "Failed to initialize multi-GPU manager");
                    return false;
                }
                multi_gpu_manager_->setResultCallback(result_callback);
            } else {
                // Single specific GPU
                mining_config.device_id = config_.gpu_ids[0];
            }
        } else {
            LOG_WARN("POOL", "No GPU configuration specified, defaulting to GPU 0");
            return false;
        }

        // Initialize single GPU mining system if not using multi-GPU
        if (!multi_gpu_manager_) {
            // Apply user config if available
            if (config_.mining_config) {
                auto user_mining_config = static_cast<const MiningConfig *>(config_.mining_config);
                ConfigUtils::applyMiningConfig(*user_mining_config, mining_config);
            }
            mining_system_ = std::make_unique<MiningSystem>(mining_config);

            // Set user config before initialize
            if (config_.mining_config) {
                mining_system_->setUserConfig(config_.mining_config);
            }

            if (!mining_system_->initialize()) {
                LOG_ERROR("POOL", "Failed to initialize mining system");
                return false;
            }
            mining_system_->setResultCallback(result_callback);
        }

        // Connect to pool
        if (!pool_client_->connect()) {
            LOG_ERROR("POOL", "Failed to connect to pool");
            return false;
        }

        running_ = true;

        // Start worker threads
        mining_thread_           = std::thread(&PoolMiningSystem::mining_loop, this);
        share_scanner_thread_    = std::thread(&PoolMiningSystem::share_scanner_loop, this);
        stats_reporter_thread_   = std::thread(&PoolMiningSystem::stats_reporter_loop, this);
        share_submission_thread_ = std::thread(&PoolMiningSystem::share_submission_loop, this);

        return true;
    }

    void PoolMiningSystem::share_submission_loop()
    {
        // Share submission doesn't need GPU context as it only sends network messages
        LOG_INFO("SHARE_SUBMIT", "Share submission loop started");
        while (running_.load()) {
            std::unique_lock lock(share_mutex_);
            share_cv_.wait_for(lock, std::chrono::seconds(1),
                               [this] { return !share_queue_.empty() || !running_.load(); });

            if (!running_.load())
                break;
            while (!share_queue_.empty()) {
                Share share = share_queue_.front();
                share_queue_.pop();
                lock.unlock();

                // Send via pool client
                if (pool_client_ && pool_client_->is_connected()) {
                    pool_client_->submit_share(share);

                    std::lock_guard stats_lock(stats_mutex_);
                    stats_.shares_submitted++;
                }

                lock.lock();
            }
        }

        LOG_INFO("SHARE_SUBMIT", "Share submission loop stopped");
    }

    void PoolMiningSystem::stop()
    {
        if (!running_.load()) {
            return;
        }

        LOG_INFO("POOL", "Stopping pool mining system...");

        running_       = false;
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

        if (share_submission_thread_.joinable()) {
            share_submission_thread_.join();
        }

        LOG_INFO("POOL", "Pool mining system stopped");
    }

    void PoolMiningSystem::mining_loop()
    {
        LOG_INFO("MINING", "Mining loop started");
        // CRITICAL: Get the correct GPU device ID
        int device_id = 0;
        if (multi_gpu_manager_) {
            // For multi-GPU, each worker thread handles its own context
            // We don't need to set it here
            LOG_INFO("MINING", "Using multi-GPU manager, context handled by workers");
        } else if (mining_system_) {
            // Get device ID from the mining system's config
            device_id = mining_system_->getConfig().device_id;
        } else if (!config_.gpu_ids.empty()) {
            // Use first GPU from the list
            device_id = config_.gpu_ids[0];
        }
        // Only set GPU context if not using multi-GPU manager
        if (!multi_gpu_manager_) {
            gpuError_t err = gpuSetDevice(device_id);
            if (err != gpuSuccess) {
                LOG_ERROR("MINING", "Failed to set GPU device ", device_id,
                          " in mining thread: ", gpuGetErrorString(err));
                return;
            }
            // Synchronize to ensure device is ready
            err = gpuDeviceSynchronize();
            if (err != gpuSuccess) {
                LOG_ERROR("MINING", "Device ", device_id, " not ready in mining thread: ", gpuGetErrorString(err));
                return;
            }
            LOG_INFO("MINING", "Mining thread GPU context set to device ", device_id);
        }

        while (running_.load()) {
            // Wait for a job to be available
            {
                std::unique_lock lock(job_mutex_);
                job_cv_.wait(lock, [this] {
                    return (current_job_.has_value() && current_mining_job_.has_value() && mining_active_.load()) ||
                           !running_.load();
                });
            }

            if (!running_.load()) {
                LOG_INFO("MINING", "Mining loop stopping (shutdown requested)");
                break;
            }

            if (!mining_active_.load()) {
                LOG_DEBUG("MINING", "Mining not active, waiting...");
                continue;
            }

            // Store the job version we're about to mine
            uint64_t mining_job_version = job_version_.load();

            try {
                // Verify GPU context if not using multi-GPU
                if (!multi_gpu_manager_) {
                    int current_device;
                    gpuGetDevice(&current_device);
                    if (current_device != device_id) {
                        LOG_WARN("MINING", "GPU context switched from ", device_id, " to ", current_device,
                                 ", resetting");
                        gpuError_t err = gpuSetDevice(device_id);
                        if (err != gpuSuccess) {
                            LOG_ERROR("MINING", "Failed to reset GPU context to device ", device_id, ": ",
                                      gpuGetErrorString(err));
                            continue;
                        }
                    }
                }
                // Get current job data and version
                MiningJob job_copy{};
                {
                    std::lock_guard lock(job_mutex_);
                    if (!current_mining_job_.has_value()) {
                        LOG_ERROR("MINING", "No mining job available!");
                        continue;
                    }
                    job_copy = *current_mining_job_;
                }

                // Create a condition function that ALSO checks if job version changed
                auto should_continue = [this, mining_job_version]() -> bool {
                    // Stop if job version changed!
                    if (job_version_.load() != mining_job_version) {
                        LOG_INFO("MINING", "Job version changed, stopping current mining");
                        return false;
                    }

                    return mining_active_.load() && running_.load() && pool_client_ && pool_client_->is_connected() &&
                           pool_client_->is_authenticated();
                };

                const uint64_t starting_nonce = global_nonce_offset_.load();
                LOG_INFO("MINING", "Starting mining with job version ", mining_job_version, " from nonce offset ",
                         starting_nonce);

                // CRITICAL: Update the job with the correct version BEFORE starting mining
                uint64_t final_nonce = starting_nonce;

                if (multi_gpu_manager_) {
                    multi_gpu_manager_->updateJobLive(job_copy, mining_job_version);
                    // Pass the global nonce offset atomic reference
                    multi_gpu_manager_->runMiningInterruptibleWithOffset(job_copy, should_continue,
                                                                         global_nonce_offset_);
                    // For multi-GPU, the global_nonce_offset_ is updated by the workers directly
                    final_nonce = global_nonce_offset_.load();
                } else if (mining_system_) {
                    mining_system_->updateJobLive(job_copy, mining_job_version);
                    // Use the method that returns final nonce
                    final_nonce =
                        mining_system_->runMiningLoopInterruptibleWithOffset(job_copy, should_continue, starting_nonce);

                    // CRITICAL: Update global offset with where we stopped
                    global_nonce_offset_.store(final_nonce);
                }

                LOG_DEBUG("MINING", "Mining stopped for job version ", mining_job_version, " at nonce offset ",
                          final_nonce);
            } catch (const std::exception &e) {
                LOG_ERROR("MINING", "Mining loop exception: ", e.what());

                // On error, try to recover by resetting GPU context (only if not multi-GPU)
                if (!multi_gpu_manager_) {
                    gpuSetDevice(device_id);
                    gpuGetLastError();  // Clear any errors
                }

                // Small delay before retry
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // If we exit the mining loop (due to job change or disconnect),
            // we'll wait for the next job
            if (!running_.load()) {
                break;
            }
        }

        mining_active_ = false;
        LOG_INFO("MINING", "Mining loop exited, final nonce position: ", global_nonce_offset_.load());
    }

    void PoolMiningSystem::reset_nonce_counter()
    {
        LOG_INFO("POOL", "Resetting nonce counter to 1");
        global_nonce_offset_.store(1);
    }

    void PoolMiningSystem::share_scanner_loop()
    {
        LOG_INFO("SCANNER", "Share scanner loop started");
        // Get the correct GPU device ID
        int device_id         = 0;
        bool need_gpu_context = false;
        if (multi_gpu_manager_) {
            // Multi-GPU manager handles contexts
            LOG_INFO("SCANNER", "Using multi-GPU manager, context handled by workers");
        } else if (mining_system_) {
            device_id        = mining_system_->getConfig().device_id;
            need_gpu_context = true;
        } else if (!config_.gpu_ids.empty()) {
            device_id        = config_.gpu_ids[0];
            need_gpu_context = true;
        }
        // Set GPU context if needed
        if (need_gpu_context) {
            gpuError_t err = gpuSetDevice(device_id);
            if (err != gpuSuccess) {
                LOG_ERROR("SCANNER", "Failed to set GPU device ", device_id,
                          " in scanner thread: ", gpuGetErrorString(err));
                return;
            }

            LOG_INFO("SCANNER", "Scanner thread GPU context set to device ", device_id);
        }

        while (running_.load()) {
            {
                std::unique_lock lock(results_mutex_);
                results_cv_.wait_for(lock, std::chrono::milliseconds(config_.share_scan_interval_ms),
                                     [this] { return !current_mining_results_.empty() || !running_.load(); });
            }
            if (!running_.load())
                break;
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

    void PoolMiningSystem::stats_reporter_loop()
    {
        // Stats reporter might need GPU context for temperature monitoring in the future
        int device_id         = 0;
        bool need_gpu_context = false;
        if (!multi_gpu_manager_ && mining_system_) {
            device_id        = mining_system_->getConfig().device_id;
            need_gpu_context = true;
        } else if (!multi_gpu_manager_ && !config_.gpu_ids.empty()) {
            device_id        = config_.gpu_ids[0];
            need_gpu_context = true;
        }

        if (need_gpu_context) {
            gpuError_t err = gpuSetDevice(device_id);
            if (err != gpuSuccess) {
                LOG_ERROR("STATS", "Failed to set GPU device ", device_id,
                          " in stats thread: ", gpuGetErrorString(err));
            } else {
                LOG_INFO("STATS", "Stats thread GPU context set to device ", device_id);
            }
        }

        auto last_hashrate_report  = std::chrono::steady_clock::now();
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
                    if (auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
                        elapsed.count() > 0) {
                        report.hashrate = stats_.hashrate;  // Updated by update_stats()
                    }
                    report.gpu_count = config_.gpu_ids.empty() ? 1 : config_.gpu_ids.size();
                } else if (mining_system_) {
                    const auto mining_stats = mining_system_->getStats();
                    report.hashrate         = mining_stats.hash_rate;
                    report.gpu_count        = 1;
                }

                report.shares_submitted = stats_.shares_submitted;
                report.shares_accepted  = stats_.shares_accepted;
                report.uptime_seconds   = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

                // Add GPU stats
                nlohmann::json gpu_stats;
                if (multi_gpu_manager_) {
                    for (uint32_t i = 0; i < report.gpu_count; i++) {
                        gpu_stats["gpu_" + std::to_string(i)]["hashrate"]    = report.hashrate / report.gpu_count;
                        gpu_stats["gpu_" + std::to_string(i)]["temperature"] = 0;  // TODO: Add temp monitoring
                    }
                } else {
                    gpu_stats["gpu_0"]["hashrate"]    = report.hashrate;
                    gpu_stats["gpu_0"]["temperature"] = 0;  // TODO: Add temp monitoring
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

    void PoolMiningSystem::scan_for_shares()
    {
        std::vector<MiningResult> results_to_process;
        std::string current_job_id;
        uint32_t job_difficulty_bits;
        uint64_t expected_job_version;

        // Swap pending results to process them
        {
            std::lock_guard lock(pending_results_mutex_);
            if (pending_results_.empty()) {
                return;
            }
            results_to_process.swap(pending_results_);
        }

        // Get current job info - USE CURRENT DIFFICULTY
        {
            std::lock_guard lock(job_mutex_);
            if (!current_job_.has_value()) {
                LOG_WARN("SHARE", "No current job, discarding ", results_to_process.size(), " results");
                return;
            }
            current_job_id = current_job_->job_id;

            // Use the current difficulty instead of job's original difficulty
            job_difficulty_bits = current_difficulty_.load();

            expected_job_version = job_version_.load();
        }

        if (current_job_id.empty()) {
            return;
        }

        LOG_DEBUG("SHARE", "Scanning ", results_to_process.size(), " results for job ", current_job_id, " version ",
                  expected_job_version, " with difficulty ", job_difficulty_bits, " bits");

        // Submit shares that meet difficulty AND are from current job
        int valid_shares = 0;
        int stale_shares = 0;

        for (const auto &result : results_to_process) {
            // Double-check job version
            if (result.job_version != expected_job_version) {
                stale_shares++;
                LOG_WARN("SHARE", "Discarding stale result from job version ", result.job_version,
                         " (current: ", expected_job_version, ")");
                continue;
            }

            // Use current difficulty for share validation
            if (result.matching_bits >= job_difficulty_bits) {
                valid_shares++;
                submit_share(result);
            }
        }

        if (stale_shares > 0) {
            LOG_WARN("SHARE", "Discarded ", stale_shares, " stale results from old job versions");
        }

        if (valid_shares > 0) {
            LOG_DEBUG("SHARE", "Submitted ", valid_shares, " valid shares for job ", current_job_id);
        }
    }

    void PoolMiningSystem::setup_mining_result_callback()
    {
        if (!mining_system_) {
            LOG_ERROR("SETUP", "Mining system not initialized");
            return;
        }
        auto result_callback = [this](const std::vector<MiningResult> &results) {
            LOG_DEBUG("CALLBACK", "Mining result callback triggered with ", results.size(), " results");
            if (results.empty()) {
                LOG_DEBUG("CALLBACK", "Empty results vector received");
                return;
            }
            // Log details of each result
            for (size_t i = 0; i < results.size(); i++) {
                const auto &result = results[i];
                LOG_DEBUG("CALLBACK", "Result ", i, ": nonce=0x", std::hex, result.nonce, std::dec,
                          ", bits=", result.matching_bits, ", difficulty_score=", result.difficulty_score);
                // Log the hash
                std::string hash_str;
                for (int j = 0; j < 5; j++) {
                    char buf[9];
                    snprintf(buf, sizeof(buf), "%08x", result.hash[j]);
                    hash_str += buf;
                    if (j < 4)
                        hash_str += " ";
                }
                LOG_DEBUG("CALLBACK", "  Hash: ", hash_str);
            }
            // Store results for share scanning
            {
                std::lock_guard lock(results_mutex_);
                const size_t old_size = current_mining_results_.size();
                current_mining_results_.insert(current_mining_results_.end(), results.begin(), results.end());
                LOG_DEBUG("CALLBACK", "Stored results, total count: ", old_size, " -> ",
                          current_mining_results_.size());
            }
            // Notify share scanner
            results_cv_.notify_one();
        };

        mining_system_->setResultCallback(result_callback);
        LOG_INFO("SETUP", "Mining result callback configured with enhanced debugging");
    }

    void PoolMiningSystem::process_mining_results(const std::vector<MiningResult> &results)
    {
        LOG_DEBUG("SHARE", "Processing ", results.size(), " mining results");

        // Get current job difficulty and version
        uint32_t current_job_difficulty;
        uint64_t expected_job_version;
        std::string current_job_id;
        {
            std::lock_guard lock(job_mutex_);
            if (current_job_.has_value()) {
                // Use current difficulty instead of job's original difficulty
                current_job_difficulty = current_difficulty_.load();
                expected_job_version   = job_version_.load();
                current_job_id         = current_job_->job_id;
            } else {
                LOG_ERROR("SHARE", "No current job, discarding results");
                return;
            }
        }

        LOG_DEBUG("SHARE", "Current job: ", current_job_id, ", requires difficulty: ", current_job_difficulty,
                  ", version: ", expected_job_version, "results: ", results.size());

        // Filter results based on current job difficulty and version
        std::vector<MiningResult> filtered_results;
        int version_mismatches = 0;

        for (const auto &result : results) {
            // Check job version to avoid stale shares
            if (result.job_version != expected_job_version) {
                version_mismatches++;
                LOG_DEBUG("SHARE", "Result has job version ", result.job_version, " but expected ",
                          expected_job_version, " - STALE");
                continue;
            }

            if (result.matching_bits >= current_job_difficulty) {
                filtered_results.push_back(result);
                LOG_DEBUG("SHARE", "Result meets difficulty: ", result.matching_bits, " bits, nonce: 0x", std::hex,
                          result.nonce, std::dec, ", version: ", result.job_version);
            } else {
                LOG_DEBUG("SHARE", "Result does not meet difficulty: ", result.matching_bits,
                          " bits (required: ", current_job_difficulty, ")");
            }
        }

        if (version_mismatches > 0) {
            LOG_WARN("SHARE", "Discarded ", version_mismatches, " results with wrong job version");
        }

        if (!filtered_results.empty()) {
            LOG_DEBUG("SHARE", "Found ", filtered_results.size(), " results meeting difficulty ",
                      current_job_difficulty, " (out of ", results.size(), " total)");

            // Use double buffer to avoid race conditions
            {
                std::lock_guard lock(pending_results_mutex_);

                // Limit pending results to prevent memory issues
                if (pending_results_.size() + filtered_results.size() > MAX_PENDING_RESULTS) {
                    LOG_WARN("SHARE", "Pending results buffer full, dropping oldest results");
                    const size_t to_remove = (pending_results_.size() + filtered_results.size()) - MAX_PENDING_RESULTS;
                    pending_results_.erase(pending_results_.begin(), pending_results_.begin() + to_remove);
                }

                pending_results_.insert(pending_results_.end(), filtered_results.begin(), filtered_results.end());
            }
            results_cv_.notify_one();
        }
    }

    void PoolMiningSystem::submit_share(const MiningResult &result)
    {
        std::lock_guard lock(job_mutex_);
        if (!current_job_.has_value()) {
            LOG_ERROR("SHARE", "No current job for share submission");
            return;
        }

        Share share;
        share.job_id        = current_job_->job_id;
        share.nonce         = result.nonce;
        share.hash          = hash_to_hex(result.hash);
        share.matching_bits = result.matching_bits;
        share.found_time    = std::chrono::steady_clock::now();

        LOG_TRACE("SHARE", "Submitting share for job ", share.job_id, ", nonce: 0x", std::hex, share.nonce, std::dec,
                  ", bits: ", share.matching_bits, ", hash: ", share.hash, ", result version: ", result.job_version,
                  ", current version: ", job_version_.load());

        // Check share queue size
        {
            std::lock_guard queue_lock(share_mutex_);
            if (share_queue_.size() >= MAX_SHARE_QUEUE_SIZE) {
                LOG_WARN("SHARE", "Share queue full (", share_queue_.size(), " shares), dropping oldest share");
                share_queue_.pop();
            }
            share_queue_.push(share);
        }
        share_cv_.notify_one();
    }

    MiningJob PoolMiningSystem::convert_to_mining_job(const JobMessage &job_msg)
    {
        MiningJob mining_job{};
        // The prefix_data now contains the unique salted preimage for this worker
        const auto prefix_bytes = Utils::hex_to_bytes(job_msg.prefix_data);
        const auto target_bytes = Utils::hex_to_bytes(job_msg.target_pattern);
        if (target_bytes.size() != 20) {
            LOG_ERROR("POOL", "Invalid target pattern size: ", target_bytes.size());
            return mining_job;
        }

        // Clear the base message
        std::memset(mining_job.base_message, 0, 32);

        // Copy the salted preimage to base message
        if (!prefix_bytes.empty()) {
            const size_t copy_size = std::min(prefix_bytes.size(), static_cast<size_t>(32));
            std::memcpy(mining_job.base_message, prefix_bytes.data(), copy_size);
            LOG_DEBUG("POOL", "Using salted preimage of ", copy_size, " bytes");
        }

        // Handle suffix data if present (usually empty for salted preimages)
        if (!job_msg.suffix_data.empty()) {
            const auto suffix_bytes = Utils::hex_to_bytes(job_msg.suffix_data);
            if (!suffix_bytes.empty() && prefix_bytes.size() + suffix_bytes.size() <= 32) {
                const size_t suffix_offset = prefix_bytes.size();
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
        mining_job.difficulty   = job_msg.target_difficulty;
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
        for (const unsigned int i : mining_job.target_hash) {
            char buf[9];
            snprintf(buf, sizeof(buf), "%08x", i);
            target_hex_verify += buf;
        }

        LOG_DEBUG("POOL", "Target pattern set to: ", target_hex_verify);

        return mining_job;
    }

    void PoolMiningSystem::update_mining_job(const PoolJob &pool_job)
    {
        // Increment job version
        const uint64_t new_version = job_version_.fetch_add(1) + 1;

        // Create new mining job
        MiningJob new_mining_job = convert_to_mining_job(pool_job.job_data);

        // Log first 16 bytes of base message (salted preimage) in hex
        std::string base_msg_hex;
        for (int i = 0; i < 16; i++) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", new_mining_job.base_message[i]);
            base_msg_hex += buf;
        }

        LOG_DEBUG("POOL", "New job base message (first 16 bytes): ", base_msg_hex, "...");

        // Only stop mining if we're actually mining
        if (mining_active_.load()) {
            LOG_INFO("POOL", "Stopping current mining for job update");
            mining_active_ = false;

            // Stop the mining system SYNCHRONOUSLY
            if (multi_gpu_manager_) {
                multi_gpu_manager_->stopMining();
                multi_gpu_manager_->sync();
            } else if (mining_system_) {
                mining_system_->stopMining();
                mining_system_->sync();
            }

            // Give time for mining threads to notice the stop
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // Update job data atomically
        {
            std::lock_guard lock(job_mutex_);

            // Clear ALL result buffers
            {
                std::lock_guard results_lock(pending_results_mutex_);
                pending_results_.clear();
                pending_results_.shrink_to_fit();
            }
            {
                std::lock_guard results_lock(results_mutex_);
                current_mining_results_.clear();
                current_mining_results_.shrink_to_fit();
            }
            {
                std::lock_guard share_lock(share_mutex_);
                std::queue<Share> empty;
                std::swap(share_queue_, empty);
            }

            // Update job data
            current_job_               = pool_job;
            current_mining_job_        = new_mining_job;
            current_job_id_for_mining_ = pool_job.job_id;
            current_difficulty_        = pool_job.job_data.target_difficulty;

            LOG_DEBUG("POOL", "Job data updated to ", pool_job.job_id, " with version ", new_version);
        }

        // Clear GPU state if we have a mining system
        if (mining_system_) {
            mining_system_->resetState();
        }

        mining_active_ = true;

        // Signal the mining loop that a job is available
        job_cv_.notify_all();
    }

    PoolMiningSystem::PoolMiningStats PoolMiningSystem::get_stats() const
    {
        std::lock_guard lock(stats_mutex_);

        auto stats = stats_;

        // Update uptime
        stats.uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);

        // Calculate success rate
        if (stats.shares_submitted > 0) {
            stats.share_success_rate =
                static_cast<double>(stats.shares_accepted) / static_cast<double>(stats.shares_submitted);
        }

        // Get connection status
        if (pool_client_) {
            stats.connected     = pool_client_->is_connected();
            stats.authenticated = stats.connected && !stats.worker_id.empty();
        }

        // Log current bit difficulty for debugging
        LOG_DEBUG("STATS", "Current difficulty: ", stats.current_difficulty, " bits");

        return stats;
    }

    void PoolMiningSystem::cleanup_mining_system()
    {
        if (mining_system_) {
            mining_system_.reset();
        }
        if (multi_gpu_manager_) {
            multi_gpu_manager_.reset();
        }
    }

    void PoolMiningSystem::update_stats()
    {
        std::lock_guard lock(stats_mutex_);
        if (mining_system_) {
            const auto mining_stats = mining_system_->getStats();
            stats_.hashrate         = mining_stats.hash_rate;
            stats_.total_hashes     = mining_stats.hashes_computed;
        } else if (multi_gpu_manager_) {
            stats_.hashrate = multi_gpu_manager_->getTotalHashRate();
            // For multi-GPU, estimate total hashes from nonce progress
            stats_.total_hashes = global_nonce_offset_.load() - 1;
        }

        // Update current difficulty from the actual job
        stats_.current_difficulty = current_difficulty_.load();

        // Calculate uptime
        const auto now = std::chrono::steady_clock::now();
        stats_.uptime  = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        // Calculate share success rate
        if (stats_.shares_submitted > 0) {
            stats_.share_success_rate = static_cast<double>(stats_.shares_accepted) / stats_.shares_submitted;
        } else {
            stats_.share_success_rate = 0.0;
        }

        // Log nonce progress periodically
        static uint64_t last_logged_nonce = 0;
        if (const uint64_t current_nonce = global_nonce_offset_.load();
            current_nonce - last_logged_nonce > 1000000000) {
            // Log every billion nonces
            LOG_DEBUG("POOL", "Nonce progress: ", current_nonce, " (", (current_nonce / 1000000000.0), " billion)");

            last_logged_nonce = current_nonce;
        }
    }

    void PoolMiningSystem::handle_reconnect()
    {
        LOG_INFO("POOL", "Handling reconnection...");

        // Stop mining temporarily
        mining_active_ = false;

        // Clear current job
        {
            std::lock_guard lock(job_mutex_);
            current_job_.reset();
            current_mining_job_.reset();
        }

        // Clear pending shares
        {
            std::lock_guard lock(share_mutex_);
            std::queue<Share> empty;
            std::swap(share_queue_, empty);
        }

        // Pool client will handle the actual reconnection
    }

    void PoolMiningSystem::adjust_local_difficulty()
    {
        // TODO
    }

    // IPoolEventHandler implementations
    void PoolMiningSystem::on_connected()
    {
        LOG_INFO("POOL", Color::GREEN, "Connected to mining pool", Color::RESET);

        std::lock_guard lock(stats_mutex_);
        stats_.connected = true;
    }

    void PoolMiningSystem::on_disconnected(const std::string &reason)
    {
        LOG_WARN("POOL", Color::RED, "Disconnected from pool: ", reason, Color::RESET);

        // CRITICAL: Stop mining immediately
        mining_active_ = false;

        // Stop the actual mining kernels
        if (multi_gpu_manager_) {
            multi_gpu_manager_->stopMining();
            multi_gpu_manager_->sync();
        } else if (mining_system_) {
            mining_system_->stopMining();
            mining_system_->sync();
        }

        // Clear current job to prevent mining with stale data
        {
            std::lock_guard lock(job_mutex_);
            current_job_.reset();
            current_mining_job_.reset();
        }

        job_cv_.notify_all();

        std::lock_guard lock(stats_mutex_);
        stats_.connected     = false;
        stats_.authenticated = false;
    }

    void PoolMiningSystem::on_error(ErrorCode code, const std::string &message)
    {
        LOG_ERROR("POOL", "Pool error (", static_cast<int>(code), "): ", message);
    }

    void PoolMiningSystem::on_authenticated(const std::string &worker_id)
    {
        LOG_INFO("POOL", Color::GREEN, "Authenticated as worker: ", worker_id, Color::RESET);

        std::lock_guard lock(stats_mutex_);
        stats_.authenticated = true;
        stats_.worker_id     = worker_id;
    }

    void PoolMiningSystem::on_auth_failed(ErrorCode code, const std::string &reason)
    {
        LOG_ERROR("POOL", Color::RED, "Authentication failed: ", reason, Color::RESET);

        std::lock_guard lock(stats_mutex_);
        stats_.authenticated = false;

        // Stop mining on auth failure
        mining_active_ = false;
    }

    void PoolMiningSystem::on_new_job(const PoolJob &job)
    {
        LOG_INFO("POOL", "New job received: ", job.job_id, " (difficulty: ", job.job_data.target_difficulty, " bits)",
                 " - Continuing from nonce: ", global_nonce_offset_.load());

        // Update current difficulty immediately
        current_difficulty_.store(job.job_data.target_difficulty);

        // Check if this is the same job ID we already have
        {
            std::lock_guard lock(job_mutex_);
            if (current_job_.has_value() && current_job_->job_id == job.job_id) {
                LOG_DEBUG("POOL", "Received duplicate job ", job.job_id, " - ignoring");
                return;
            }
        }

        // Always treat pool jobs as clean jobs that require full restart
        // BUT we keep the nonce position!
        if (mining_active_.load()) {
            LOG_DEBUG("POOL", "Stopping mining for new job (nonce will continue from ", global_nonce_offset_.load(),
                      ")");

            // Stop mining
            mining_active_ = false;

            // Stop GPU mining immediately
            if (multi_gpu_manager_) {
                multi_gpu_manager_->stopMining();
                multi_gpu_manager_->sync();
            } else if (mining_system_) {
                mining_system_->stopMining();
                mining_system_->sync();
            }

            // Wait for mining to fully stop
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        // Update the job
        update_mining_job(job);
    }

    void PoolMiningSystem::on_job_cancelled(const std::string &job_id)
    {
        LOG_DEBUG("POOL", "Job cancelled: ", job_id);

        std::lock_guard lock(job_mutex_);
        if (current_job_.has_value() && current_job_->job_id == job_id) {
            current_job_.reset();
            current_mining_job_.reset();
            mining_active_ = false;
            job_cv_.notify_all();
        }
    }

    void PoolMiningSystem::on_share_accepted(const ShareResultMessage &result)
    {
        // Log formatted difficulty if available
        if (!result.difficulty_info.is_null() && result.difficulty_info.contains("formatted_difficulty")) {
            LOG_INFO("POOL", Color::BRIGHT_GREEN,
                     "Share accepted! Difficulty: ", result.difficulty_info["formatted_difficulty"].get<std::string>(),
                     Color::RESET);
        } else {
            LOG_INFO("POOL", Color::BRIGHT_GREEN, "Share accepted!", Color::RESET);
        }

        // Update stats
        {
            std::lock_guard lock(stats_mutex_);
            stats_.shares_accepted++;

            // Track best share if we have bits information
            if (result.bits_matched > 0 && result.bits_matched > stats_.best_share_bits) {
                stats_.best_share_bits = result.bits_matched;
                LOG_INFO("POOL", Color::BRIGHT_CYAN, "*** NEW BEST SHARE: ", result.bits_matched, " bits! ***",
                         Color::RESET);
            }

            // Track share times for vardiff
            share_times_.push_back(std::chrono::steady_clock::now());
            if (share_times_.size() > 100) {
                share_times_.pop_front();
            }

            if (!result.message.empty()) {
                // Check for special messages
                if (result.message.find("High-value contribution") != std::string::npos) {
                    LOG_INFO("POOL", Color::BRIGHT_YELLOW, "*** HIGH-VALUE EPOCH CONTRIBUTION FOUND! ***",
                             Color::RESET);
                }
            }

            // Log share value if provided
            if (result.share_value > 0) {
                LOG_DEBUG("POOL", "Share value: ", result.share_value);
            }
        }
    }

    void PoolMiningSystem::on_share_rejected(const ShareResultMessage &result)
    {
        LOG_WARN("POOL", Color::RED, "Share rejected: ", result.message, Color::RESET);

        // Update stats
        {
            std::lock_guard lock(stats_mutex_);
            stats_.shares_rejected++;
        }
    }

    void PoolMiningSystem::on_difficulty_changed(const uint32_t new_difficulty)
    {
        LOG_INFO("POOL", Color::BRIGHT_MAGENTA, "Difficulty adjusted to: ", new_difficulty, " bits", " (",
                 DifficultyConverter::formatDifficulty(DifficultyConverter::bitsToScaledDifficulty(new_difficulty)),
                 ")", " - Nonce continues from: ", global_nonce_offset_.load(), Color::RESET);

        // Update the atomic difficulty
        current_difficulty_ = new_difficulty;

        // Update stats
        {
            std::lock_guard lock(stats_mutex_);
            stats_.current_difficulty = new_difficulty;
            stats_.current_bits       = new_difficulty;
        }

        // CRITICAL: Update the current job's difficulty if we have one
        {
            std::lock_guard lock(job_mutex_);
            if (current_job_.has_value()) {
                // Update the job message difficulty
                current_job_->job_data.target_difficulty = new_difficulty;

                // Also update the mining job difficulty
                if (current_mining_job_.has_value()) {
                    current_mining_job_->difficulty = new_difficulty;

                    LOG_DEBUG("POOL", "Updated mining job difficulty to ", new_difficulty,
                              " bits (nonce continues from ", global_nonce_offset_.load(), ")");
                }
            }
        }
    }

    void PoolMiningSystem::on_pool_status(const PoolStatusMessage &status)
    {
        LOG_INFO("POOL", "Pool status - Workers: ", status.connected_workers,
                 ", Hashrate: ", status.total_hashrate / 1e9, " GH/s", ", Epoch: #", status.current_epoch_number,
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

        std::lock_guard lock(stats_mutex_);
    }

    // MultiPoolManager implementation
    MultiPoolManager::MultiPoolManager() = default;

    MultiPoolManager::~MultiPoolManager()
    {
        stop_mining();
    }

    void MultiPoolManager::add_pool(const std::string &name, const PoolConfig &config, const int priority)
    {
        std::lock_guard lock(mutex_);

        PoolEntry entry;
        entry.name     = name;
        entry.config   = config;
        entry.priority = priority;
        entry.enabled  = true;

        pools_.push_back(std::move(entry));
        sort_pools_by_priority();
    }

    void MultiPoolManager::remove_pool(const std::string &name)
    {
        std::lock_guard lock(mutex_);

        std::erase_if(pools_, [&name](const PoolEntry &entry) { return entry.name == name; });
    }

    void MultiPoolManager::set_pool_priority(const std::string &name, const int priority)
    {
        std::lock_guard lock(mutex_);

        const auto it = std::ranges::find_if(pools_, [&name](const PoolEntry &entry) { return entry.name == name; });

        if (it != pools_.end()) {
            it->priority = priority;
            sort_pools_by_priority();
        }
    }

    void MultiPoolManager::enable_pool(const std::string &name, const bool enable)
    {
        std::lock_guard lock(mutex_);

        if (const auto it =
                std::ranges::find_if(pools_, [&name](const PoolEntry &entry) { return entry.name == name; });
            it != pools_.end()) {
            it->enabled = enable;
        }
    }

    bool MultiPoolManager::start_mining(const PoolMiningSystem::Config &mining_config)
    {
        if (running_.load()) {
            return true;
        }

        std::lock_guard lock(mutex_);

        if (pools_.empty()) {
            LOG_ERROR("MULTI_POOL", "No pools configured");
            return false;
        }

        base_mining_config_ = mining_config;
        running_            = true;

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

    void MultiPoolManager::stop_mining()
    {
        if (!running_.load()) {
            return;
        }

        running_ = false;

        // Stop current mining
        {
            std::lock_guard lock(mutex_);
            for (auto &pool : pools_) {
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

    std::string MultiPoolManager::get_active_pool() const
    {
        std::lock_guard lock(mutex_);
        return active_pool_;
    }

    std::map<std::string, PoolMiningSystem::PoolMiningStats> MultiPoolManager::get_all_stats() const
    {
        std::lock_guard lock(mutex_);

        std::map<std::string, PoolMiningSystem::PoolMiningStats> all_stats;

        for (const auto &pool : pools_) {
            if (pool.mining_system) {
                all_stats[pool.name] = pool.mining_system->get_stats();
            }
        }

        return all_stats;
    }

    void MultiPoolManager::failover_monitor()
    {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));

            std::lock_guard lock(mutex_);

            // Check if current pool is still connected
            auto it =
                std::ranges::find_if(pools_, [this](const PoolEntry &entry) { return entry.name == active_pool_; });

            if (it != pools_.end() && it->mining_system) {
                if (const auto stats = it->mining_system->get_stats(); !stats.connected || !stats.authenticated) {
                    LOG_WARN("MULTI_POOL", "Pool ", active_pool_, " disconnected, failing over...");
                    try_next_pool();
                }
            }
        }
    }

    bool MultiPoolManager::try_next_pool()
    {
        // Stop current pool if any
        for (auto &pool : pools_) {
            if (pool.name == active_pool_ && pool.mining_system) {
                pool.mining_system->stop();
                pool.mining_system.reset();
            }
        }

        // Try each pool in priority order
        for (auto &pool : pools_) {
            if (!pool.enabled) {
                continue;
            }

            LOG_INFO("MULTI_POOL", "Trying pool: ", pool.name);

            // Create mining config for this pool
            auto config        = base_mining_config_;
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

    void MultiPoolManager::sort_pools_by_priority()
    {
        std::ranges::sort(pools_, [](const PoolEntry &a, const PoolEntry &b) { return a.priority < b.priority; });
    }
}  // namespace MiningPool
