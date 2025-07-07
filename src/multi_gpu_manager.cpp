#include "multi_gpu_manager.hpp"
#include "utilities.hpp"
#ifdef USE_HIP
#include "gpu_architecture.hpp"
#endif
#include <iostream>
#include <iomanip>
#include <algorithm>

MultiGPUManager::MultiGPUManager() {
    start_time_ = std::chrono::steady_clock::now();
}

MultiGPUManager::~MultiGPUManager() {
    shutdown_ = true;
    for (auto &worker: workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
    }
}

bool MultiGPUManager::initialize(const std::vector<int> &gpu_ids) {
    std::cout << "\nInitializing Multi-GPU Mining System\n";
    std::cout << "=====================================\n";

    for (int gpu_id: gpu_ids) {
        auto worker = std::make_unique<GPUWorker>();
        worker->device_id = gpu_id;

        // Get device properties first
        gpuDeviceProp props;
        gpuError_t err = gpuGetDeviceProperties(&props, gpu_id);
        if (err != gpuSuccess) {
            std::cerr << "Failed to get properties for GPU " << gpu_id << "\n";
            continue;
        }

        std::cout << "\nInitializing GPU " << gpu_id << ": " << props.name << "\n";

#ifdef USE_HIP
        // Check for AMD GPU issues
        AMDArchitecture arch = AMDGPUDetector::detectArchitecture(props);
        AMDArchParams arch_params = AMDGPUDetector::getArchitectureParams(arch);

        std::cout << "  Architecture: " << arch_params.arch_name << " (" << props.gcnArchName << ")\n";
        std::cout << "  Wavefront size: " << arch_params.wavefront_size << "\n";

        if (AMDGPUDetector::hasKnownIssues(arch, props.name)) {
            std::cout << "WARNING: GPU " << gpu_id << " has known compatibility issues.\n";

            // Get ROCm version
            int version;
            if (hipRuntimeGetVersion(&version) == hipSuccess) {
                int major = version / 10000000;
                int minor = (version % 10000000) / 100000;
                int patch = (version % 100000) / 100;
                std::cout << "Current ROCm version: " << major << "." << minor << "." << patch << "\n";

                if (version < 50700000 && arch == AMDArchitecture::RDNA3) {
                    std::cout << "RDNA3 requires ROCm 5.7 or later. ";

                    // Ask user whether to skip or try anyway
                    std::cout << "Skip this GPU? (Recommended) [Y/n]: ";
                    std::string response;
                    std::getline(std::cin, response);
                    if (response.empty() || response[0] == 'Y' || response[0] == 'y') {
                        std::cout << "Skipping GPU " << gpu_id << "\n";
                        continue;
                    }
                    std::cout << "Attempting to initialize with reduced settings...\n";
                }
            }
        }
#endif

        // Create mining system for this GPU
        MiningSystem::Config config;
        config.device_id = gpu_id;
        config.num_streams = 4;
        config.threads_per_block = DEFAULT_THREADS_PER_BLOCK;
        config.use_pinned_memory = true;
        config.result_buffer_size = 256;
#ifdef USE_HIP
        // Apply architecture-specific configuration for problematic GPUs
        if (arch == AMDArchitecture::RDNA3) {
            std::cout << "Applying RDNA3-specific workarounds...\n";
            config.num_streams = 2; // Reduce streams
            config.blocks_per_stream = 256; // Start conservative
            config.threads_per_block = 128; // Smaller workgroups
        }
#endif

        worker->mining_system = std::make_unique<MiningSystem>(config);
        // Try to initialize with error handling
        try {
            if (!worker->mining_system->initialize()) {
                std::cerr << "Failed to initialize GPU " << gpu_id << "\n";
                continue;
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception initializing GPU " << gpu_id << ": " << e.what() << "\n";
            continue;
        } catch (...) {
            std::cerr << "Unknown exception initializing GPU " << gpu_id << "\n";
            continue;
        }

        workers_.push_back(std::move(worker));
        std::cout << "Successfully initialized GPU " << gpu_id << "\n";
    }

    if (workers_.empty()) {
        std::cerr << "No GPUs were successfully initialized\n";
        return false;
    }

    std::cout << "\nSuccessfully initialized " << workers_.size() << " GPU(s) for mining\n";
    std::cout << "=====================================\n\n";

    return true;
}

uint64_t MultiGPUManager::getNextNonceBatch() {
    return global_nonce_counter_.fetch_add(NONCE_BATCH_SIZE);
}

void MultiGPUManager::workerThread(GPUWorker *worker, const MiningJob &job) {
    // Set GPU context for this thread
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        std::cerr << "[GPU " << worker->device_id << "] Failed to set device context: " << gpuGetErrorString(err) <<
                "\n";
        return;
    }

    std::cout << "[GPU " << worker->device_id << "] Worker thread started\n";

    // Set result callback for this worker
    auto worker_callback = [this, worker](const std::vector<MiningResult> &results) {
        // Update worker stats
        worker->candidates_found += results.size();
        // Check for new best and forward to global callback
        for (const auto &result: results) {
            if (result.matching_bits > worker->best_match_bits) {
                worker->best_match_bits = result.matching_bits;
                // Check if this is a global best
                if (global_best_tracker_.isNewBest(result.matching_bits)) {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - start_time_
                    );
                    std::cout << "\n[GPU " << worker->device_id << " - NEW BEST!] Time: " << elapsed.count() << "s\n";
                    std::cout << "  Nonce: 0x" << std::hex << result.nonce << std::dec << "\n";
                    std::cout << "  Matching bits: " << result.matching_bits << "\n";
                    std::cout << "  Hash: ";
                    for (int j = 0; j < 5; j++) {
                        std::cout << std::hex << std::setw(8) << std::setfill('0')
                                << result.hash[j];
                        if (j < 4) std::cout << " ";
                    }
                    std::cout << std::dec << "\n\n";
                }
            }
        }

        // Forward to global callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (result_callback_) {
                result_callback_(results);
            }
        }
    };

    worker->mining_system->setResultCallback(worker_callback);

    // For problematic GPUs, add extra error handling
    bool has_errors = false;
    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;

    // Get initial nonce batch
    uint64_t current_nonce_base = getNextNonceBatch();
    uint64_t nonces_used_in_batch = 0;

    // Run until shutdown signal - NO TIME LIMIT
    while (!shutdown_ && !has_errors) {
        try {
            // Create job with current nonce offset
            MiningJob worker_job = job;
            worker_job.nonce_offset = current_nonce_base + nonces_used_in_batch;

            // Run a single kernel batch with error handling
            auto kernel_start = std::chrono::steady_clock::now();
            uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);
            if (hashes_this_round == 0) {
                // Fallback estimation
                auto config = worker->mining_system->getConfig();
                hashes_this_round = static_cast<uint64_t>(config.blocks_per_stream) * config.threads_per_block *
                                    NONCES_PER_THREAD;
            }

            // Update worker stats
            worker->hashes_computed += hashes_this_round;
            nonces_used_in_batch += hashes_this_round;

            // Check if we need a new nonce batch
            if (nonces_used_in_batch >= NONCE_BATCH_SIZE * 0.9) {
                current_nonce_base = getNextNonceBatch();
                nonces_used_in_batch = 0;
            }

            // Reset error counter on success
            consecutive_errors = 0;
        } catch (const std::exception &e) {
            std::cerr << "[GPU " << worker->device_id << "] Error: " << e.what() << "\n";
            consecutive_errors++;

            if (consecutive_errors >= max_consecutive_errors) {
                std::cerr << "[GPU " << worker->device_id << "] Too many consecutive errors, stopping worker\n";
                has_errors = true;
                break;
            }

            // Wait a bit before retrying
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (has_errors) {
        std::cerr << "[GPU " << worker->device_id << "] Worker thread stopped due to errors\n";
    } else {
        std::cout << "[GPU " << worker->device_id << "] Worker thread finished (shutdown signal received)\n";
    }
}

void MultiGPUManager::workerThreadInterruptible(GPUWorker *worker, const MiningJob &job,
                                                std::function<bool()> should_continue) {
    // Set GPU context for this thread
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        std::cerr << "[GPU " << worker->device_id << "] Failed to set device context: " << gpuGetErrorString(err) <<
                "\n";
        return;
    }

    std::cout << "[GPU " << worker->device_id << "] Worker thread started (interruptible mode)\n";

    // Set result callback for this worker
    auto worker_callback = [this, worker](const std::vector<MiningResult> &results) {
        // Update worker stats
        worker->candidates_found += results.size();
        // Check for new best and forward to global callback
        for (const auto &result: results) {
            if (result.matching_bits > worker->best_match_bits) {
                worker->best_match_bits = result.matching_bits;
                // Check if this is a global best
                if (global_best_tracker_.isNewBest(result.matching_bits)) {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - start_time_
                    );
                    std::cout << "\n[GPU " << worker->device_id << " - NEW BEST!] Time: " << elapsed.count() << "s\n";
                    std::cout << "  Nonce: 0x" << std::hex << result.nonce << std::dec << "\n";
                    std::cout << "  Matching bits: " << result.matching_bits << "\n";
                    std::cout << "  Hash: ";
                    for (int j = 0; j < 5; j++) {
                        std::cout << std::hex << std::setw(8) << std::setfill('0')
                                << result.hash[j];
                        if (j < 4) std::cout << " ";
                    }
                    std::cout << std::dec << "\n\n";
                }
            }
        }

        // Forward to global callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (result_callback_) {
                result_callback_(results);
            }
        }
    };

    worker->mining_system->setResultCallback(worker_callback);

    // For problematic GPUs, add extra error handling
    bool has_errors = false;
    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;

    // Get initial nonce batch
    uint64_t current_nonce_base = getNextNonceBatch();
    uint64_t nonces_used_in_batch = 0;

    // Run until shutdown signal OR connection lost
    while (!shutdown_ && !has_errors && should_continue()) {
        try {
            // Create job with current nonce offset
            MiningJob worker_job = job;
            worker_job.nonce_offset = current_nonce_base + nonces_used_in_batch;

            // Run a single kernel batch with error handling
            auto kernel_start = std::chrono::steady_clock::now();
            uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);
            if (hashes_this_round == 0) {
                // Fallback estimation
                auto config = worker->mining_system->getConfig();
                hashes_this_round = static_cast<uint64_t>(config.blocks_per_stream) * config.threads_per_block *
                                    NONCES_PER_THREAD;
            }

            // Update worker stats
            worker->hashes_computed += hashes_this_round;
            nonces_used_in_batch += hashes_this_round;

            // Check if we need a new nonce batch
            if (nonces_used_in_batch >= NONCE_BATCH_SIZE * 0.9) {
                current_nonce_base = getNextNonceBatch();
                nonces_used_in_batch = 0;
            }

            // Reset error counter on success
            consecutive_errors = 0;
        } catch (const std::exception &e) {
            std::cerr << "[GPU " << worker->device_id << "] Error: " << e.what() << "\n";
            consecutive_errors++;

            if (consecutive_errors >= max_consecutive_errors) {
                std::cerr << "[GPU " << worker->device_id << "] Too many consecutive errors, stopping worker\n";
                has_errors = true;
                break;
            }

            // Wait a bit before retrying
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!should_continue()) {
        std::cout << "[GPU " << worker->device_id << "] Worker thread stopped (connection lost)\n";
    } else if (has_errors) {
        std::cerr << "[GPU " << worker->device_id << "] Worker thread stopped due to errors\n";
    } else {
        std::cout << "[GPU " << worker->device_id << "] Worker thread finished (shutdown signal received)\n";
    }
}

void MultiGPUManager::runMiningInterruptible(const MiningJob &job, std::function<bool()> should_continue) {
    std::cout << "\nStarting interruptible multi-GPU mining on " << workers_.size() << " device(s)\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Mining will stop when connection is lost\n";
    std::cout << "=====================================\n\n";

    // Store difficulty for stats calculation
    current_difficulty_ = job.difficulty;

    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    // Reset all worker stats
    for (auto &worker: workers_) {
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }

    // Start worker threads with connection checking
    for (auto &worker: workers_) {
        worker->worker_thread = std::make_unique<std::thread>(
            &MultiGPUManager::workerThreadInterruptible, this, worker.get(), job, should_continue
        );
    }

    // Monitor progress with proper synchronization
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_total_hashes = 0;

    // Monitor until shutdown signal OR connection lost
    while (!shutdown_ && should_continue()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        // Calculate combined stats with atomic reads
        uint64_t total_hashes = 0;
        uint64_t total_candidates = 0;
        uint32_t best_bits = global_best_tracker_.getBestBits();

        std::vector<double> gpu_rates;
        std::vector<uint64_t> gpu_hashes;
        // Collect stats from each GPU
        for (const auto &worker: workers_) {
            uint64_t gpu_hash_count = worker->hashes_computed.load();
            gpu_hashes.push_back(gpu_hash_count);
            total_hashes += gpu_hash_count;
            total_candidates += worker->candidates_found.load();

            // Calculate per-GPU rate
            double gpu_rate = 0.0;
            if (elapsed.count() > 0) {
                gpu_rate = static_cast<double>(gpu_hash_count) / elapsed.count() / 1e9;
            }
            gpu_rates.push_back(gpu_rate);
        }

        // Calculate rates
        uint64_t hash_diff = total_hashes - last_total_hashes;
        auto interval = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        double instant_rate = 0.0;
        double average_rate = 0.0;
        if (interval.count() > 0) {
            instant_rate = static_cast<double>(hash_diff) / interval.count() / 1e9;
        }
        if (elapsed.count() > 0) {
            average_rate = static_cast<double>(total_hashes) / elapsed.count() / 1e9;
        }

        // Print status line
        std::cout << "\r[" << elapsed.count() << "s] "
                << "Rate: " << std::fixed << std::setprecision(2)
                << instant_rate << " GH/s"
                << " (avg: " << average_rate << " GH/s) | "
                << "Best: " << best_bits << " bits | "
                << "GPUs: ";

        // Show per-GPU rates
        for (size_t i = 0; i < gpu_rates.size(); i++) {
            if (i > 0) std::cout << "+";
            std::cout << std::fixed << std::setprecision(1) << gpu_rates[i];
        }

        std::cout << " | Total: " << std::fixed << std::setprecision(3)
                << static_cast<double>(total_hashes) / 1e12
                << " TH" << std::flush;

        last_update = now;
        last_total_hashes = total_hashes;
    }

    // Connection lost or shutdown requested
    if (!should_continue()) {
        std::cout << "\n\n[MULTI-GPU] Pool connection lost - stopping all workers...\n";
    }

    // Signal shutdown and wait for all workers
    shutdown_ = true;
    std::cout << "\n\nShutting down workers...\n";

    for (auto &worker: workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
    }

    printCombinedStats();
}

void MultiGPUManager::runMining(const MiningJob &job) {
    std::cout << "\nStarting infinite multi-GPU mining on " << workers_.size() << " device(s)\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Press Ctrl+C to stop mining\n";
    std::cout << "=====================================\n\n";

    // Store difficulty for stats calculation
    current_difficulty_ = job.difficulty;

    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    // Reset all worker stats
    for (auto &worker: workers_) {
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }

    // Start worker threads
    for (auto &worker: workers_) {
        worker->worker_thread = std::make_unique<std::thread>(
            &MultiGPUManager::workerThread, this, worker.get(), job // No duration parameter
        );
    }

    // Monitor progress with proper synchronization
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_total_hashes = 0;

    // Monitor until shutdown signal
    while (!shutdown_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        // Calculate combined stats with atomic reads
        uint64_t total_hashes = 0;
        uint64_t total_candidates = 0;
        uint32_t best_bits = global_best_tracker_.getBestBits();

        std::vector<double> gpu_rates;
        std::vector<uint64_t> gpu_hashes;
        // Collect stats from each GPU
        for (const auto &worker: workers_) {
            uint64_t gpu_hash_count = worker->hashes_computed.load();
            gpu_hashes.push_back(gpu_hash_count);
            total_hashes += gpu_hash_count;
            total_candidates += worker->candidates_found.load();

            // Calculate per-GPU rate
            double gpu_rate = 0.0;
            if (elapsed.count() > 0) {
                gpu_rate = static_cast<double>(gpu_hash_count) / elapsed.count() / 1e9;
            }
            gpu_rates.push_back(gpu_rate);
        }

        // Calculate rates
        uint64_t hash_diff = total_hashes - last_total_hashes;
        auto interval = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        double instant_rate = 0.0;
        double average_rate = 0.0;
        if (interval.count() > 0) {
            instant_rate = static_cast<double>(hash_diff) / interval.count() / 1e9;
        }
        if (elapsed.count() > 0) {
            average_rate = static_cast<double>(total_hashes) / elapsed.count() / 1e9;
        }

        // Print status line
        std::cout << "\r[" << elapsed.count() << "s] "
                << "Rate: " << std::fixed << std::setprecision(2)
                << instant_rate << " GH/s"
                << " (avg: " << average_rate << " GH/s) | "
                << "Best: " << best_bits << " bits | "
                << "GPUs: ";

        // Show per-GPU rates
        for (size_t i = 0; i < gpu_rates.size(); i++) {
            if (i > 0) std::cout << "+";
            std::cout << std::fixed << std::setprecision(1) << gpu_rates[i];
        }

        std::cout << " | Total: " << std::fixed << std::setprecision(3)
                << static_cast<double>(total_hashes) / 1e12
                << " TH" << std::flush;

        last_update = now;
        last_total_hashes = total_hashes;
    }

    // Signal shutdown and wait for all workers
    shutdown_ = true;
    std::cout << "\n\nShutting down workers...\n";

    for (auto &worker: workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
    }

    printCombinedStats();
}

void MultiGPUManager::printCombinedStats() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );

    uint64_t total_hashes = 0;
    uint64_t total_candidates = 0;
    uint32_t best_bits = global_best_tracker_.getBestBits();

    std::cout << "\n=== Multi-GPU Mining Results ===\n";
    std::cout << "=====================================\n";

    // Per-GPU stats
    for (size_t i = 0; i < workers_.size(); i++) {
        const auto &worker = workers_[i];
        uint64_t gpu_hashes = worker->hashes_computed.load();
        uint64_t gpu_candidates = worker->candidates_found.load();
        uint32_t gpu_best = worker->best_match_bits.load();
        double gpu_rate = 0.0;
        if (elapsed.count() > 0) {
            gpu_rate = static_cast<double>(gpu_hashes) / elapsed.count() / 1e9;
        }

        // Get GPU name
        gpuDeviceProp props;

        gpuError_t err = gpuGetDeviceProperties(&props, worker->device_id);
        if (err != gpuSuccess) {
            std::cerr << "Failed to get device properties for GPU " << worker->device_id
                      << ": " << gpuGetErrorString(err) << std::endl;
            continue; // Skip this GPU
        }

        std::cout << "GPU " << worker->device_id << " (" << props.name << "):\n";
        std::cout << "  Total Hashes: " << std::fixed << std::setprecision(3)
                << static_cast<double>(gpu_hashes) / 1e9 << " GH\n";
        std::cout << "  Hash Rate: " << std::fixed << std::setprecision(2)
                << gpu_rate << " GH/s\n";
        std::cout << "  Best Match: " << gpu_best << " bits\n";
        std::cout << "  Candidates: " << gpu_candidates << "\n";
        if (gpu_hashes > 0 && gpu_candidates > 0) {
            double efficiency = 100.0 * gpu_candidates * std::pow(2.0, current_difficulty_) / gpu_hashes;
            std::cout << "  Efficiency: " << std::fixed << std::setprecision(4)
                    << efficiency << "%\n";
        }
        std::cout << "\n";

        total_hashes += gpu_hashes;
        total_candidates += gpu_candidates;
    }

    std::cout << "=====================================\n";
    std::cout << "Combined Statistics:\n";
    std::cout << "  Platform: " << getGPUPlatformName() << "\n";
    std::cout << "  Total GPUs: " << workers_.size() << "\n";
    std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " << std::fixed << std::setprecision(3)
            << static_cast<double>(total_hashes) / 1e12 << " TH\n";
    if (elapsed.count() > 0) {
        std::cout << "  Combined Rate: " << std::fixed << std::setprecision(2)
                << static_cast<double>(total_hashes) / elapsed.count() / 1e9 << " GH/s\n";
    }

    std::cout << "  Best Match: " << best_bits << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates << "\n";

    if (total_hashes > 0 && total_candidates > 0) {
        double global_efficiency = 100.0 * total_candidates * std::pow(2.0, current_difficulty_) / total_hashes;
        std::cout << "  Global Efficiency: " << std::scientific << std::setprecision(2)
                << global_efficiency << "%\n";
    }
    std::cout << "=====================================\n";
}
