#include "multi_gpu_manager.hpp"
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
    stopMining();
}

bool MultiGPUManager::initialize(const std::vector<int>& gpu_ids) {
    std::cout << "\nInitializing Multi-GPU Mining System\n";
    std::cout << "=====================================\n";
    
    for (int gpu_id : gpu_ids) {
        auto worker = std::make_unique<GPUWorker>();
        worker->device_id = gpu_id;
        
        // Get device properties
        gpuDeviceProp props;
        gpuError_t err = gpuGetDeviceProperties(&props, gpu_id);
        if (err != gpuSuccess) {
            std::cerr << "Failed to get properties for GPU " << gpu_id << "\n";
            continue;
        }
        
        std::cout << "\nInitializing GPU " << gpu_id << ": " << props.name << "\n";
        
#ifdef USE_HIP
        // AMD-specific initialization
        AMDArchitecture arch = AMDGPUDetector::detectArchitecture(props);
        AMDArchParams arch_params = AMDGPUDetector::getArchitectureParams(arch, props);
        
        std::cout << "  Architecture: " << AMDGPUDetector::getArchitectureName(arch) 
                  << " (" << props.gcnArchName << ")\n";
        std::cout << "  Wave size: " << arch_params.wave_size << "\n";
        std::cout << "  Max waves per CU: " << arch_params.waves_per_cu << "\n";
        std::cout << "  Compute units: " << props.multiProcessorCount << "\n";
        
        if (AMDGPUDetector::hasKnownIssues(arch, props.name)) {
            std::cout << "WARNING: GPU " << gpu_id << " has known compatibility issues.\n";
            
            // Check ROCm version
            int version;
            if (hipRuntimeGetVersion(&version) == hipSuccess) {
                int major = version / 10000000;
                int minor = (version % 10000000) / 100000;
                int patch = (version % 100000) / 100;
                std::cout << "Current ROCm version: " << major << "." << minor << "." << patch << "\n";
                
                if (version < 50700000 && arch == AMDArchitecture::RDNA3) {
                    std::cout << "RDNA3 requires ROCm 5.7 or later. Skipping GPU " << gpu_id << "\n";
                    continue;
                }
            }
        }
#endif
        
        // Create mining system configuration
        MiningSystem::Config config;
        config.device_id = gpu_id;
        config.num_streams = 8;
        config.threads_per_block = DEFAULT_THREADS_PER_BLOCK;
        config.use_pinned_memory = true;
        config.result_buffer_size = 1024;
        
#ifdef USE_HIP
        // Apply architecture-specific configuration
        if (arch == AMDArchitecture::RDNA3) {
            std::cout << "Applying RDNA3-specific optimizations...\n";
            config.num_streams = 2;
            config.blocks_per_stream = 256;
            config.threads_per_block = 128;
        }
#endif
        
        // Create and initialize mining system
        worker->mining_system = std::make_unique<MiningSystem>(config);
        
        try {
            if (!worker->mining_system->initialize()) {
                std::cerr << "Failed to initialize GPU " << gpu_id << "\n";
                continue;
            }
        } catch (const std::exception& e) {
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

void MultiGPUManager::stopMining() {
    if (shutdown_.load()) {
        return;
    }
    
    LOG_INFO("MULTI_GPU", "Stopping all mining operations...");
    
    // Signal shutdown
    shutdown_ = true;
    
    // Stop all worker mining systems
    for (auto& worker : workers_) {
        if (worker->mining_system) {
            worker->mining_system->stopMining();
        }
    }
    
    // Wait for monitor thread
    //if (monitor_thread_ && monitor_thread_->joinable()) {
    //    monitor_thread_->join();
    //}
    
    // Wait for all worker threads
    waitForWorkers();
    
    LOG_INFO("MULTI_GPU", "All mining operations stopped");
}

void MultiGPUManager::waitForWorkers() {
    for (auto& worker : workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
        worker->active = false;
    }
}

uint64_t MultiGPUManager::getNextNonceBatch() {
    return global_nonce_counter_.fetch_add(NONCE_BATCH_SIZE);
}

void MultiGPUManager::processWorkerResults(GPUWorker* worker, const std::vector<MiningResult>& results) {
    // Update worker stats
    worker->candidates_found += results.size();
    
    // Check for new best
    for (const auto& result : results) {
        if (result.matching_bits > worker->best_match_bits) {
            worker->best_match_bits = result.matching_bits;
            
            // Check if this is a global best
            if (global_best_tracker_.isNewBest(result.matching_bits)) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time_
                );
                
                std::cout << "\n[GPU " << worker->device_id << " - NEW BEST!] Time: " 
                         << elapsed.count() << "s\n";
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
}

void MultiGPUManager::workerThread(GPUWorker* worker, const MiningJob& job) {
    // Set GPU context for this thread
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, 
                  " - Failed to set device context: ", gpuGetErrorString(err));
        return;
    }
    
    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread started");
    worker->active = true;
    
    // Set result callback
    auto worker_callback = [this, worker](const std::vector<MiningResult>& results) {
        processWorkerResults(worker, results);
    };
    worker->mining_system->setResultCallback(worker_callback);
    
    // Error handling
    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;
    
    // Get initial nonce batch
    uint64_t current_nonce_base = getNextNonceBatch();
    uint64_t nonces_used_in_batch = 0;
    
    // Run until shutdown
    while (!shutdown_ && consecutive_errors < max_consecutive_errors) {
        try {
            // Create job with current nonce offset
            MiningJob worker_job = job;
            worker_job.nonce_offset = current_nonce_base + nonces_used_in_batch;
            
            // Run a single kernel batch
            uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);
            
            if (hashes_this_round == 0) {
                // Fallback estimation
                auto config = worker->mining_system->getConfig();
                hashes_this_round = static_cast<uint64_t>(config.blocks_per_stream) * 
                                   config.threads_per_block * NONCES_PER_THREAD;
            }
            
            // Update stats
            worker->hashes_computed += hashes_this_round;
            nonces_used_in_batch += hashes_this_round;
            
            // Check if we need a new nonce batch
            if (nonces_used_in_batch >= NONCE_BATCH_SIZE * 0.9) {
                current_nonce_base = getNextNonceBatch();
                nonces_used_in_batch = 0;
            }
            
            // Reset error counter on success
            consecutive_errors = 0;
            
        } catch (const std::exception& e) {
            LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Error: ", e.what());
            consecutive_errors++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    worker->active = false;
    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker thread finished");
}

void MultiGPUManager::workerThreadInterruptibleWithOffset(GPUWorker* worker,
                                                         const MiningJob& job,
                                                         std::function<bool()> should_continue,
                                                         std::atomic<uint64_t>& shared_nonce_counter) {
    // Set GPU context for this thread
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id,
                  " - Failed to set device context: ", gpuGetErrorString(err));
        return;
    }

    LOG_INFO("MULTI_GPU", "GPU ", worker->device_id,
             " - Worker thread started (continuous nonce mode)");
    worker->active = true;

    // Set result callback
    auto worker_callback = [this, worker](const std::vector<MiningResult>& results) {
        processWorkerResults(worker, results);
    };
    worker->mining_system->setResultCallback(worker_callback);

    // Error handling
    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;

    // Each GPU will grab nonce batches from the shared counter
    const uint64_t gpu_batch_size = NONCE_BATCH_SIZE;

    // Run until shutdown OR should_continue returns false
    while (!shutdown_ && consecutive_errors < max_consecutive_errors && should_continue()) {
        try {
            // Atomically get next nonce batch from shared counter
            uint64_t batch_start = shared_nonce_counter.fetch_add(gpu_batch_size);

            LOG_TRACE("MULTI_GPU", "GPU ", worker->device_id,
                      " - Got nonce batch starting at: ", batch_start);

            // Create job with this GPU's nonce batch
            MiningJob worker_job = job;
            worker_job.nonce_offset = batch_start;

            // Instead of runSingleBatch, use the continuous nonce method
            // We'll run a limited batch to allow checking should_continue frequently
            uint64_t nonces_to_process = gpu_batch_size;
            uint64_t nonces_processed = 0;

            // Process the batch in smaller chunks to allow responsive stopping
            const uint64_t chunk_size = gpu_batch_size / 10; // Process in 10 chunks

            while (nonces_processed < nonces_to_process && should_continue() && !shutdown_) {
                // Update job offset for this chunk
                worker_job.nonce_offset = batch_start + nonces_processed;

                // Run a single kernel batch
                uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);

                if (hashes_this_round == 0) {
                    // Fallback estimation
                    auto config = worker->mining_system->getConfig();
                    hashes_this_round = static_cast<uint64_t>(config.blocks_per_stream) *
                                       config.threads_per_block * NONCES_PER_THREAD;
                }

                // Update stats
                worker->hashes_computed += hashes_this_round;
                nonces_processed += hashes_this_round;

                // Don't process more than our allocated batch
                if (nonces_processed >= nonces_to_process) {
                    break;
                }
            }

            // Reset error counter on success
            consecutive_errors = 0;

        } catch (const std::exception& e) {
            LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Error: ", e.what());
            consecutive_errors++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    worker->active = false;

    if (!should_continue()) {
        LOG_INFO("MULTI_GPU", "GPU ", worker->device_id,
                 " - Worker stopped (external signal) at global nonce: ",
                 shared_nonce_counter.load());
    } else if (consecutive_errors >= max_consecutive_errors) {
        LOG_ERROR("MULTI_GPU", "GPU ", worker->device_id, " - Worker stopped (too many errors)");
    } else {
        LOG_INFO("MULTI_GPU", "GPU ", worker->device_id, " - Worker finished (shutdown)");
    }
}


/*void MultiGPUManager::monitorThread(std::function<bool()> should_continue) {
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_total_hashes = 0;
    
    while (!shutdown_ && should_continue()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
        
        // Calculate combined stats
        uint64_t total_hashes = 0;
        uint64_t total_candidates = 0;
        uint32_t best_bits = global_best_tracker_.getBestBits();
        
        std::vector<double> gpu_rates;
        for (const auto& worker : workers_) {
            if (!worker->active) continue;
            
            uint64_t gpu_hash_count = worker->hashes_computed.load();
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
        
        // Print status
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
}*/

void MultiGPUManager::runMining(const MiningJob& job) {
    current_difficulty_ = job.difficulty;
    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();
    
    // Reset all worker stats
    for (auto& worker : workers_) {
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }
    
    // Start worker threads
    for (auto& worker : workers_) {
        worker->worker_thread = std::make_unique<std::thread>(
            &MultiGPUManager::workerThread, this, worker.get(), job
        );
    }
    
    // Start monitor thread
    //monitor_thread_ = std::make_unique<std::thread>(
    //    &MultiGPUManager::monitorThread, this, []() { return true; }
    //);
    
    // Wait for workers to finish (only on shutdown)
    waitForWorkers();
    
    // Wait for monitor thread
    //if (monitor_thread_ && monitor_thread_->joinable()) {
    //    monitor_thread_->join();
    //}
    
    //printCombinedStats();
}

void MultiGPUManager::runMiningInterruptibleWithOffset(const MiningJob& job,
                                                      std::function<bool()> should_continue,
                                                      std::atomic<uint64_t>& global_nonce_offset) {
    current_difficulty_ = job.difficulty;
    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    // Reset all worker stats
    for (auto& worker : workers_) {
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }

    LOG_INFO("MULTI_GPU", "Starting multi-GPU mining from nonce offset: ",
             global_nonce_offset.load());

    // Start worker threads with shared nonce counter
    for (auto& worker : workers_) {
        worker->worker_thread = std::make_unique<std::thread>(
            &MultiGPUManager::workerThreadInterruptibleWithOffset,
            this, worker.get(), job, should_continue, std::ref(global_nonce_offset)
        );
    }

    // Wait for workers to finish
    waitForWorkers();

    if (!should_continue()) {
        LOG_INFO("MULTI_GPU", "External stop signal received - all workers stopped at nonce: ",
                 global_nonce_offset.load());
    }
}

void MultiGPUManager::sync() const {
    for (const auto& worker : workers_) {
        if (worker->mining_system && worker->active) {
            worker->mining_system->sync();
        }
    }
}

void MultiGPUManager::updateJobLive(const MiningJob& job, uint64_t job_version) const {
    // Update job on all active GPUs
    for (const auto& worker : workers_) {
        if (worker->mining_system && worker->active) {
            worker->mining_system->updateJobLive(job, job_version);
        }
    }
    
    LOG_INFO("MULTI_GPU", "Updated job on all active GPUs to version ", job_version);
}

double MultiGPUManager::getTotalHashRate() const {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );
    
    if (elapsed.count() == 0) return 0.0;
    
    uint64_t total_hashes = 0;
    for (const auto& worker : workers_) {
        total_hashes += worker->hashes_computed.load();
    }
    
    return static_cast<double>(total_hashes) / elapsed.count();
}

size_t MultiGPUManager::getActiveWorkerCount() const {
    size_t count = 0;
    for (const auto& worker : workers_) {
        if (worker->active.load()) {
            count++;
        }
    }
    return count;
}

bool MultiGPUManager::allWorkersReady() const {
    for (const auto& worker : workers_) {
        if (!worker->active.load()) {
            return false;
        }
    }
    return true;
}

/*void MultiGPUManager::printCombinedStats() const {
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
        const auto& worker = workers_[i];
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
            continue;
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
    std::cout << "  Active GPUs: " << getActiveWorkerCount() << "\n";
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
}*/