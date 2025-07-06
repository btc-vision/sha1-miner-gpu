#include "../include/multi_gpu_manager.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "utilities.hpp"

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

        std::cout << "Initializing GPU " << gpu_id << ": " << props.name << "\n";

        // Create mining system for this GPU
        MiningSystem::Config config;
        config.device_id = gpu_id;
        config.num_streams = 4;
        config.threads_per_block = DEFAULT_THREADS_PER_BLOCK;
        config.use_pinned_memory = true;
        config.result_buffer_size = 256;

        worker->mining_system = std::make_unique<MiningSystem>(config);
        if (!worker->mining_system->initialize()) {
            std::cerr << "Failed to initialize GPU " << gpu_id << "\n";
            continue;
        }

        workers_.push_back(std::move(worker));
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

void MultiGPUManager::workerThread(GPUWorker *worker, const MiningJob &job, uint32_t duration_seconds) {
    // Set GPU context for this thread
    gpuError_t err = gpuSetDevice(worker->device_id);
    if (err != gpuSuccess) {
        std::cerr << "[GPU " << worker->device_id << "] Failed to set device context\n";
        return;
    }

    std::cout << "[GPU " << worker->device_id << "] Worker thread started\n";

    auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);

    while (!shutdown_ && std::chrono::steady_clock::now() < end_time) {
        // Get next nonce batch for this GPU
        uint64_t nonce_start = getNextNonceBatch();

        // Create modified job with new nonce offset
        MiningJob worker_job = job;
        worker_job.nonce_offset = nonce_start;

        // Calculate interval for this mining round
        uint32_t interval = 30; // Mine for 30 seconds at a time
        auto time_left = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - std::chrono::steady_clock::now()
        ).count();

        if (time_left <= 0) break;
        if (time_left < interval) interval = time_left;

        // Reset mining system state for this round
        worker->mining_system->resetState();

        // Run mining
        worker->mining_system->runMiningLoop(worker_job, interval);

        // Update worker stats
        auto stats = worker->mining_system->getStats();
        worker->hashes_computed += stats.hashes_computed;
        worker->candidates_found += stats.candidates_found;

        // Update best match
        if (stats.best_match_bits > worker->best_match_bits) {
            worker->best_match_bits = stats.best_match_bits;

            // Check if this is a global best
            if (global_best_tracker_.isNewBest(stats.best_match_bits)) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                std::cout << "\n[GPU " << worker->device_id << "] New global best: "
                        << stats.best_match_bits << " bits!\n" << std::flush;
            }
        }
    }

    std::cout << "[GPU " << worker->device_id << "] Worker thread finished\n";
}

void MultiGPUManager::runMining(const MiningJob &job, uint32_t duration_seconds) {
    std::cout << "\nStarting multi-GPU mining on " << workers_.size() << " device(s)\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Duration: " << duration_seconds << " seconds\n";
    std::cout << "=====================================\n\n";

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
            &MultiGPUManager::workerThread, this, worker.get(), job, duration_seconds
        );
    }

    // Monitor progress
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_total_hashes = 0;

    while (!shutdown_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        if (elapsed.count() >= duration_seconds) {
            shutdown_ = true;
            break;
        }

        // Calculate combined stats
        uint64_t total_hashes = 0;
        uint64_t total_candidates = 0;
        uint32_t best_bits = global_best_tracker_.getBestBits();

        std::vector<double> gpu_rates;
        for (const auto &worker: workers_) {
            uint64_t gpu_hashes = worker->hashes_computed.load();
            total_hashes += gpu_hashes;
            total_candidates += worker->candidates_found.load();

            // Calculate per-GPU rate
            double gpu_rate = static_cast<double>(gpu_hashes) / elapsed.count() / 1e9;
            gpu_rates.push_back(gpu_rate);
        }

        // Calculate instant and average rates
        auto interval = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        uint64_t hash_diff = total_hashes - last_total_hashes;
        double instant_rate = static_cast<double>(hash_diff) / interval.count() / 1e9;
        double average_rate = static_cast<double>(total_hashes) / elapsed.count() / 1e9;

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

        std::cout << " | Total: " << static_cast<double>(total_hashes) / 1e12
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
        double gpu_rate = static_cast<double>(gpu_hashes) / elapsed.count() / 1e9;

        // Get GPU name
        gpuDeviceProp props;
        gpuGetDeviceProperties(&props, worker->device_id);

        std::cout << "GPU " << worker->device_id << " (" << props.name << "):\n";
        std::cout << "  Hash Rate: " << std::fixed << std::setprecision(2)
                << gpu_rate << " GH/s\n";
        std::cout << "  Best Match: " << gpu_best << " bits\n";
        std::cout << "  Candidates: " << gpu_candidates << "\n";
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(4)
                << (100.0 * gpu_candidates * std::pow(2.0, best_bits) / gpu_hashes)
                << "%\n\n";

        total_hashes += gpu_hashes;
        total_candidates += gpu_candidates;
    }

    std::cout << "=====================================\n";
    std::cout << "Combined Statistics:\n";
    std::cout << "  Platform: " << getGPUPlatformName() << "\n";
    std::cout << "  Total GPUs: " << workers_.size() << "\n";
    std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " << static_cast<double>(total_hashes) / 1e12 << " TH\n";
    std::cout << "  Combined Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_hashes) / elapsed.count() / 1e9 << " GH/s\n";
    std::cout << "  Best Match: " << best_bits << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates << "\n";
    std::cout << "  Global Efficiency: " << std::scientific << std::setprecision(2)
            << (100.0 * total_candidates * std::pow(2.0, best_bits) / total_hashes)
            << "%\n";
    std::cout << "=====================================\n";
}
