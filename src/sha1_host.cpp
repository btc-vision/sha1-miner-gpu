#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>

extern std::atomic<bool> g_shutdown;

// Global state
struct MiningSystem {
    int device_id{};
    cudaDeviceProp device_props{};
    std::vector<cudaStream_t> streams;
    std::vector<ResultPool> gpu_pools;

    // Performance tracking
    std::atomic<uint64_t> total_hashes;
    std::atomic<uint64_t> total_candidates;
    std::chrono::steady_clock::time_point start_time;

    // Configuration
    int num_streams;
    int blocks_per_stream{};
    int threads_per_block{};

    // Thread management
    std::unique_ptr<std::thread> monitor_thread;
    std::mutex system_mutex;

    MiningSystem() : total_hashes(0), total_candidates(0), num_streams(4) {}

    ~MiningSystem() {
        // Ensure monitor thread is joined
        if (monitor_thread && monitor_thread->joinable()) {
            monitor_thread->join();
        }
    }
};

static MiningSystem* g_system = nullptr;

// Initialize the mining system
bool init_mining_system(int device_id) {
    if (g_system) {
        std::cerr << "Mining system already initialized\n";
        return false;
    }

    g_system = new MiningSystem();
    g_system->device_id = device_id;

    // Set device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        delete g_system;
        g_system = nullptr;
        return false;
    }

    // Get device properties
    err = cudaGetDeviceProperties(&g_system->device_props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << "\n";
        delete g_system;
        g_system = nullptr;
        return false;
    }

    // Print device info
    std::cout << "Initializing SHA-1 Near-Collision Miner\n";
    std::cout << "Device: " << g_system->device_props.name << "\n";
    std::cout << "Compute Capability: " << g_system->device_props.major << "."
              << g_system->device_props.minor << "\n";
    std::cout << "SMs: " << g_system->device_props.multiProcessorCount << "\n";
    std::cout << "Max Threads per Block: " << g_system->device_props.maxThreadsPerBlock << "\n";

    // Configure kernel launch parameters
    g_system->threads_per_block = 256;
    g_system->blocks_per_stream = g_system->device_props.multiProcessorCount * 4;

    // Create streams and allocate GPU memory
    g_system->streams.resize(g_system->num_streams);
    g_system->gpu_pools.resize(g_system->num_streams);

    for (int i = 0; i < g_system->num_streams; i++) {
        // Create stream
        err = cudaStreamCreateWithPriority(&g_system->streams[i],
                                          cudaStreamNonBlocking,
                                          i % 2); // Alternate priorities
        if (err != cudaSuccess) {
            std::cerr << "Failed to create stream " << i << ": "
                      << cudaGetErrorString(err) << "\n";
            cleanup_mining_system();
            return false;
        }

        // Allocate result pool
        ResultPool& pool = g_system->gpu_pools[i];
        pool.capacity = MAX_CANDIDATES_PER_BATCH;

        err = cudaMalloc(&pool.results, sizeof(MiningResult) * pool.capacity);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate results buffer: "
                      << cudaGetErrorString(err) << "\n";
            cleanup_mining_system();
            return false;
        }

        err = cudaMalloc(&pool.count, sizeof(uint32_t));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate count: "
                      << cudaGetErrorString(err) << "\n";
            cleanup_mining_system();
            return false;
        }

        // Initialize count to 0
        cudaMemsetAsync(pool.count, 0, sizeof(uint32_t), g_system->streams[i]);
    }

    // Set up persistent L2 cache if available (Ampere+)
    if (g_system->device_props.major >= 8) {
        const size_t l2_cache_size = g_system->device_props.l2CacheSize;
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_cache_size);
    }

    g_system->start_time = std::chrono::steady_clock::now();

    std::cout << "Mining system initialized successfully\n";
    std::cout << "Streams: " << g_system->num_streams << "\n";
    std::cout << "Blocks per Stream: " << g_system->blocks_per_stream << "\n";
    std::cout << "Threads per Block: " << g_system->threads_per_block << "\n";
    std::cout << "Total Threads: " << g_system->num_streams * g_system->blocks_per_stream *
                 g_system->threads_per_block << "\n";
    std::cout << "Hashes per Kernel: " << g_system->blocks_per_stream *
                 g_system->threads_per_block * NONCES_PER_THREAD << "\n\n";

    return true;
}

// Create a mining job
MiningJob create_mining_job(const uint8_t* message, const uint8_t* target_hash, uint32_t difficulty) {
    MiningJob job{};

    // Copy message
    std::memcpy(job.base_message, message, 32);

    // Convert target hash to uint32_t array
    for (int i = 0; i < 5; i++) {
        job.target_hash[i] = (static_cast<uint32_t>(target_hash[i*4]) << 24) |
                            (static_cast<uint32_t>(target_hash[i*4 + 1]) << 16) |
                            (static_cast<uint32_t>(target_hash[i*4 + 2]) << 8) |
                            static_cast<uint32_t>(target_hash[i*4 + 3]);
    }

    job.difficulty = difficulty;
    job.nonce_offset = 0;

    return job;
}

// Process results from GPU
int process_results(ResultPool& pool, MiningResult* host_results, int max_results) {
    if (!g_system) return 0;

    // Get result count
    uint32_t count;
    cudaMemcpy(&count, pool.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (count == 0) return 0;

    // Limit to max_results
    count = std::min(count, (uint32_t)max_results);

    // Copy results
    cudaMemcpy(host_results, pool.results, sizeof(MiningResult) * count, cudaMemcpyDeviceToHost);

    // Reset count
    cudaMemset(pool.count, 0, sizeof(uint32_t));

    // Update statistics
    g_system->total_candidates += count;

    return count;
}

// Performance monitoring thread
void performance_monitor_thread(MiningSystem* system) {
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_hashes = 0;

    while (system && !g_shutdown) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        if (!system || g_shutdown) break;

        std::lock_guard<std::mutex> lock(system->system_mutex);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - system->start_time);

        uint64_t current_hashes = system->total_hashes.load();
        uint64_t hash_diff = current_hashes - last_hashes;

        double instant_rate = hash_diff / static_cast<double>(elapsed.count()) / 1e9;
        double average_rate = current_hashes / static_cast<double>(total_elapsed.count()) / 1e9;

        std::cout << "\r[" << total_elapsed.count() << "s] "
                  << "Rate: " << std::fixed << std::setprecision(2) << instant_rate << " GH/s"
                  << " (avg: " << average_rate << " GH/s) | "
                  << "Candidates: " << system->total_candidates.load()
                  << " | Total: " << current_hashes / 1e12 << " TH"
                  << std::flush;

        last_update = now;
        last_hashes = current_hashes;
    }
}

// Cleanup
void cleanup_mining_system() {
    if (!g_system) return;

    std::cout << "\nCleaning up mining system...\n";

    // Signal shutdown
    g_shutdown = true;

    // Wait for monitor thread
    if (g_system->monitor_thread && g_system->monitor_thread->joinable()) {
        g_system->monitor_thread->join();
    }

    // Synchronize all streams
    for (const auto& stream : g_system->streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    // Free GPU memory
    for (auto& pool : g_system->gpu_pools) {
        if (pool.results) cudaFree(pool.results);
        if (pool.count) cudaFree(pool.count);
    }

    // Print final statistics
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - g_system->start_time);

    std::cout << "\nFinal Statistics:\n";
    std::cout << "Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "Total Hashes: " << g_system->total_hashes.load() / 1e12 << " trillion\n";
    std::cout << "Average Rate: " << g_system->total_hashes.load() / static_cast<double>(elapsed.count()) / 1e9 << " GH/s\n";
    std::cout << "Total Candidates: " << g_system->total_candidates.load() << "\n";

    delete g_system;
    g_system = nullptr;
}

// Helper function to run mining loop
void run_mining_loop(MiningJob job, uint32_t duration_seconds) {
    if (!g_system) {
        std::cerr << "Mining system not initialized\n";
        return;
    }

    std::cout << "Starting mining for " << duration_seconds << " seconds...\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n\n";

    // Start performance monitor
    g_system->monitor_thread = std::make_unique<std::thread>(performance_monitor_thread, g_system);

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);

    uint64_t nonce_offset = 0;
    std::vector<MiningResult> results(MAX_CANDIDATES_PER_BATCH);

    int stream_idx = 0;

    while (std::chrono::steady_clock::now() < end_time && !g_shutdown) {
        // Round-robin through streams
        auto& stream = g_system->streams[stream_idx];
        auto& pool = g_system->gpu_pools[stream_idx];

        // Update job nonce offset
        job.nonce_offset = nonce_offset;

        // Configure kernel launch
        KernelConfig config{};
        config.blocks = g_system->blocks_per_stream;
        config.threads_per_block = g_system->threads_per_block;
        config.shared_memory_size = 0;
        config.stream = stream;

        // Launch kernel
        launch_mining_kernel(job, pool, config);

        // Update counters
        uint64_t hashes_this_launch = config.blocks * config.threads_per_block * NONCES_PER_THREAD;
        nonce_offset += hashes_this_launch;
        g_system->total_hashes += hashes_this_launch;

        // Check for results from previous launches
        int prev_stream = (stream_idx - 1 + g_system->num_streams) % g_system->num_streams;
        if (cudaStreamQuery(g_system->streams[prev_stream]) == cudaSuccess) {
            int count = process_results(g_system->gpu_pools[prev_stream],
                                      results.data(), results.size());

            // Process any found candidates
            for (int i = 0; i < count; i++) {
                std::cout << "\n[CANDIDATE] Nonce: 0x" << std::hex << results[i].nonce
                          << " | Matching bits: " << std::dec << results[i].matching_bits
                          << " | Score: " << results[i].difficulty_score << "\n";
            }
        }

        stream_idx = (stream_idx + 1) % g_system->num_streams;
    }

    // Wait for all streams to complete
    for (auto& stream : g_system->streams) {
        cudaStreamSynchronize(stream);
    }

    // Process final results
    for (int i = 0; i < g_system->num_streams; i++) {
        const int count = process_results(g_system->gpu_pools[i], results.data(), results.size());
        for (int j = 0; j < count; j++) {
            std::cout << "\n[CANDIDATE] Nonce: 0x" << std::hex << results[j].nonce
                      << " | Matching bits: " << std::dec << results[j].matching_bits
                      << " | Score: " << results[j].difficulty_score << "\n";
        }
    }

    g_system->monitor_thread->join();
}