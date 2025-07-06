#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>
#include "sha1_miner_core.cuh"
#include "cxxsha1.hpp"
#include "job_upload_api.h"

class SHA1CollisionMiner {
private:
    struct GPUContext {
        int device_id;
        cudaStream_t stream;
        CollisionCandidate* d_candidates;
        CollisionCandidate* d_filtered;
        uint32_t* d_candidate_count;
        uint32_t* d_filtered_count;
        uint32_t* d_best_distance;

        // Host pinned memory for async transfers
        CollisionCandidate* h_candidates;
        uint32_t* h_counts;

        size_t candidates_size;
        dim3 grid_size;
        dim3 block_size;
    };

    std::vector<GPUContext> gpu_contexts;
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<uint32_t> best_global_distance{160};
    std::atomic<bool> collision_found{false};
    std::atomic<uint64_t> total_candidates{0};

    MiningJob current_job;
    uint32_t target_difficulty;
    uint64_t nonce_counter{0};

    std::chrono::steady_clock::time_point start_time;

    void initialize_gpu(GPUContext& ctx, int device_id) {
        ctx.device_id = device_id;

        cudaSetDevice(device_id);
        cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking);

        // Allocate device memory
        ctx.candidates_size = sizeof(CollisionCandidate) * CANDIDATES_RING_SIZE;
        cudaMalloc(&ctx.d_candidates, ctx.candidates_size);
        cudaMalloc(&ctx.d_filtered, ctx.candidates_size / 4);
        cudaMalloc(&ctx.d_candidate_count, sizeof(uint32_t));
        cudaMalloc(&ctx.d_filtered_count, sizeof(uint32_t));
        cudaMalloc(&ctx.d_best_distance, sizeof(uint32_t));

        // Allocate pinned host memory
        cudaMallocHost(&ctx.h_candidates, ctx.candidates_size / 4);
        cudaMallocHost(&ctx.h_counts, sizeof(uint32_t) * 3);

        // Determine optimal launch configuration
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        ctx.block_size = dim3(256);
        int blocks = prop.multiProcessorCount * 32;
        ctx.grid_size = dim3(blocks);

        std::cout << "GPU " << device_id << " (" << prop.name << "): "
                  << blocks << " blocks x 256 threads = "
                  << (blocks * 256) << " threads\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0)
                  << " GB\n";
    }

    void upload_job(const MiningJob& job) {
        // Convert target hash to proper format for upload_new_job
        uint32_t target[5];
        for (int i = 0; i < 5; i++) {
            target[i] = job.target_hash[i];
        }

        // Upload using the existing API
        upload_new_job((const uint8_t*)job.base_msg, target);

        // Also upload to our custom structure
        for (auto& ctx : gpu_contexts) {
            cudaSetDevice(ctx.device_id);
            cudaMemcpyToSymbolAsync(g_job, &job, sizeof(MiningJob), 0,
                                   cudaMemcpyHostToDevice, ctx.stream);
        }

        // Initialize bit masks in constant memory
        uint32_t bit_masks[32];
        for (int i = 0; i < 32; i++) {
            bit_masks[i] = 1u << (31 - i);
        }

        for (auto& ctx : gpu_contexts) {
            cudaSetDevice(ctx.device_id);
            cudaMemcpyToSymbolAsync(g_bit_masks, bit_masks, sizeof(bit_masks), 0,
                                   cudaMemcpyHostToDevice, ctx.stream);
        }
    }

    void mine_on_gpu(GPUContext& ctx) {
        cudaSetDevice(ctx.device_id);

        while (!collision_found.load()) {
            // Reset counts
            cudaMemsetAsync(ctx.d_candidate_count, 0, sizeof(uint32_t), ctx.stream);
            cudaMemsetAsync(ctx.d_filtered_count, 0, sizeof(uint32_t), ctx.stream);
            cudaMemsetAsync(ctx.d_best_distance, 160, sizeof(uint32_t), ctx.stream);

            // Get nonce range for this batch
            uint64_t batch_start = nonce_counter.fetch_add(1ull << 32);
            current_job.nonce_start = batch_start;
            current_job.nonce_range = 1ull << 32;

            // Update job nonce range
            cudaMemcpyToSymbolAsync(g_job, &current_job.nonce_start,
                                   sizeof(uint64_t) * 2,
                                   offsetof(MiningJob, nonce_start),
                                   cudaMemcpyHostToDevice, ctx.stream);

            // Launch mining kernel
            uint64_t total_threads = ctx.grid_size.x * ctx.block_size.x;
            sha1_collision_mine_kernel<<<ctx.grid_size, ctx.block_size, 0, ctx.stream>>>(
                ctx.d_candidates, ctx.d_candidate_count, ctx.d_best_distance, total_threads
            );

            // Get initial results
            cudaMemcpyAsync(ctx.h_counts, ctx.d_candidate_count, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost, ctx.stream);
            cudaMemcpyAsync(ctx.h_counts + 1, ctx.d_best_distance, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost, ctx.stream);

            cudaStreamSynchronize(ctx.stream);

            uint32_t candidate_count = ctx.h_counts[0];
            uint32_t best_distance = ctx.h_counts[1];

            if (candidate_count > 0) {
                total_candidates.fetch_add(candidate_count);

                // Update global best
                uint32_t current_best = best_global_distance.load();
                while (best_distance < current_best &&
                       !best_global_distance.compare_exchange_weak(current_best, best_distance)) {
                    current_best = best_global_distance.load();
                }

                // Filter candidates based on dynamic threshold
                uint32_t threshold = std::min(best_distance + 10, target_difficulty + 20);

                dim3 filter_grid((candidate_count + 255) / 256);
                filter_candidates_kernel<<<filter_grid, 256, 0, ctx.stream>>>(
                    ctx.d_candidates, ctx.d_filtered, candidate_count,
                    ctx.d_filtered_count, threshold
                );

                // Get filtered results
                cudaMemcpyAsync(ctx.h_counts + 2, ctx.d_filtered_count, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, ctx.stream);
                cudaStreamSynchronize(ctx.stream);

                uint32_t filtered_count = ctx.h_counts[2];

                if (filtered_count > 0) {
                    // Transfer filtered candidates
                    size_t transfer_size = std::min(filtered_count, CANDIDATES_RING_SIZE / 4);
                    cudaMemcpyAsync(ctx.h_candidates, ctx.d_filtered,
                                   sizeof(CollisionCandidate) * transfer_size,
                                   cudaMemcpyDeviceToHost, ctx.stream);
                    cudaStreamSynchronize(ctx.stream);

                    // Process candidates
                    process_candidates(ctx.h_candidates, transfer_size, ctx.device_id);
                }
            }

            // Update hash count
            total_hashes.fetch_add(current_job.nonce_range);
        }
    }

    void process_candidates(CollisionCandidate* candidates, size_t count, int gpu_id) {
        for (size_t i = 0; i < count; i++) {
            auto& cand = candidates[i];

            // Verify the hash using cxxsha1
            uint8_t msg[32];
            memcpy(msg, current_job.base_msg, 32);

            // Apply nonce to message
            uint32_t* msg_words = (uint32_t*)msg;
            msg_words[6] ^= (uint32_t)(cand.nonce & 0xFFFFFFFF);
            msg_words[7] ^= (uint32_t)(cand.nonce >> 32);

            // Calculate SHA-1
            SHA1 sha1;
            sha1.update(std::string((char*)msg, 32));
            std::string hash_str = sha1.final();

            // Convert hash string to binary
            uint8_t computed_hash[20];
            for (int j = 0; j < 20; j++) {
                std::string byte_str = hash_str.substr(j * 2, 2);
                computed_hash[j] = std::stoul(byte_str, nullptr, 16);
            }

            // Verify distance
            uint32_t actual_distance = 0;
            for (int j = 0; j < 20; j++) {
                uint8_t xor_val = computed_hash[j] ^ ((uint8_t*)current_job.target_hash)[j];
                actual_distance += __builtin_popcount(xor_val);
            }

            // Log near-collisions
            if (actual_distance <= target_difficulty) {
                std::cout << "\n[GPU " << gpu_id << "] NEAR-COLLISION! Distance: "
                         << actual_distance << " bits\n";
                std::cout << "  Nonce: 0x" << std::hex << cand.nonce << std::dec << "\n";
                std::cout << "  Hash: " << hash_str << "\n";

                if (actual_distance == 0) {
                    std::cout << "\n\n*** EXACT COLLISION FOUND! ***\n";
                    collision_found.store(true);
                    save_collision(cand, msg, computed_hash);
                    return;
                }

                save_candidate(cand, msg, computed_hash, actual_distance);
            }
        }
    }

    void save_collision(const CollisionCandidate& cand, const uint8_t* msg, const uint8_t* hash) {
        std::ofstream file("COLLISION_FOUND.txt", std::ios::app);

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        file << "========================================\n";
        file << "SHA-1 EXACT COLLISION FOUND!\n";
        file << "Time: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
        file << "========================================\n\n";

        file << "Nonce: 0x" << std::hex << cand.nonce << "\n\n";

        file << "Message (hex): ";
        for (int i = 0; i < 32; i++) {
            file << std::hex << std::setw(2) << std::setfill('0') << (int)msg[i];
        }
        file << "\n\n";

        file << "SHA-1 Hash: ";
        for (int i = 0; i < 20; i++) {
            file << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        file << "\n\n";

        file << "Target Hash: ";
        for (int i = 0; i < 20; i++) {
            file << std::hex << std::setw(2) << std::setfill('0')
                 << (int)((uint8_t*)current_job.target_hash)[i];
        }
        file << "\n\n";

        file.close();
    }

    void save_candidate(const CollisionCandidate& cand, const uint8_t* msg,
                       const uint8_t* hash, uint32_t distance) {
        std::ofstream file("near_collisions.txt", std::ios::app);

        file << "Distance: " << distance << " bits | ";
        file << "Nonce: 0x" << std::hex << cand.nonce << " | ";
        file << "Hash: ";
        for (int i = 0; i < 20; i++) {
            file << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        file << "\n";

        file.close();
    }

public:
    SHA1CollisionMiner(uint32_t difficulty = 50) : target_difficulty(difficulty) {
        int device_count;
        cudaGetDeviceCount(&device_count);

        std::cout << "\n+------------------------------------------+\n";
        std::cout << "|     SHA-1 Collision Miner v2.0          |\n";
        std::cout << "+------------------------------------------+\n\n";

        gpu_contexts.resize(device_count);
        for (int i = 0; i < device_count; i++) {
            initialize_gpu(gpu_contexts[i], i);
        }

        start_time = std::chrono::steady_clock::now();
    }

    ~SHA1CollisionMiner() {
        for (auto& ctx : gpu_contexts) {
            cudaSetDevice(ctx.device_id);
            cudaFree(ctx.d_candidates);
            cudaFree(ctx.d_filtered);
            cudaFree(ctx.d_candidate_count);
            cudaFree(ctx.d_filtered_count);
            cudaFree(ctx.d_best_distance);
            cudaFreeHost(ctx.h_candidates);
            cudaFreeHost(ctx.h_counts);
            cudaStreamDestroy(ctx.stream);
        }
    }

    void start_mining(const uint8_t* preimage, uint32_t difficulty_override = 0) {
        if (difficulty_override > 0) {
            target_difficulty = difficulty_override;
        }

        // Setup job
        memcpy(current_job.base_msg, preimage, 32);

        // Calculate target hash using cxxsha1
        SHA1 sha1;
        sha1.update(std::string((char*)preimage, 32));
        std::string target_hash_str = sha1.final();

        // Convert hash string to binary and uint32_t array
        uint8_t target_hash_bytes[20];
        for (int i = 0; i < 20; i++) {
            std::string byte_str = target_hash_str.substr(i * 2, 2);
            target_hash_bytes[i] = std::stoul(byte_str, nullptr, 16);
        }

        // Convert to uint32_t array for GPU
        for (int i = 0; i < 5; i++) {
            current_job.target_hash[i] = (uint32_t(target_hash_bytes[i*4]) << 24) |
                                        (uint32_t(target_hash_bytes[i*4+1]) << 16) |
                                        (uint32_t(target_hash_bytes[i*4+2]) << 8) |
                                        uint32_t(target_hash_bytes[i*4+3]);
        }

        current_job.difficulty_bits = target_difficulty;

        // Upload job to GPUs
        upload_job(current_job);

        std::cout << "Starting SHA-1 collision mining...\n";
        std::cout << "Target difficulty: " << target_difficulty << " matching bits\n";
        std::cout << "Target preimage: ";
        for (int i = 0; i < 32; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)preimage[i];
        }
        std::cout << "\n";
        std::cout << "Target SHA-1: " << target_hash_str << "\n";
        std::cout << "GPUs: " << gpu_contexts.size() << "\n\n";

        // Start mining threads
        std::vector<std::thread> mining_threads;
        for (auto& ctx : gpu_contexts) {
            mining_threads.emplace_back([this, &ctx]() { mine_on_gpu(ctx); });
        }

        // Monitor progress
        while (!collision_found.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();

            uint64_t hashes = total_hashes.load();
            double hash_rate = hashes / (double)elapsed / 1e9;
            uint64_t candidates = total_candidates.load();

            std::cout << "\r[" << elapsed << "s] "
                     << "Rate: " << std::fixed << std::setprecision(2)
                     << hash_rate << " GH/s | "
                     << "Hashes: " << (hashes / 1e12) << "T | "
                     << "Best: " << (160 - best_global_distance.load()) << " bits | "
                     << "Candidates: " << candidates << "        " << std::flush;
        }

        // Wait for threads to finish
        for (auto& t : mining_threads) {
            t.join();
        }

        // Final stats
        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();

        std::cout << "\n\nMining completed in " << total_elapsed << " seconds\n";
        std::cout << "Total hashes: " << total_hashes.load() / 1e12 << " trillion\n";
        std::cout << "Average rate: " << (total_hashes.load() / (double)total_elapsed / 1e9)
                  << " GH/s\n";
    }
};

int main(int argc, char** argv) {
    // Parse command line arguments
    uint32_t difficulty = 50;
    bool use_random = true;
    uint8_t preimage[32] = {0};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--difficulty" && i + 1 < argc) {
            difficulty = std::stoi(argv[++i]);
        } else if (arg == "--preimage" && i + 1 < argc) {
            std::string hex = argv[++i];
            if (hex.length() == 64) {
                for (int j = 0; j < 32; j++) {
                    std::string byte = hex.substr(j * 2, 2);
                    preimage[j] = std::stoul(byte, nullptr, 16);
                }
                use_random = false;
            }
        } else if (arg == "--help") {
            std::cout << "SHA-1 Collision Miner v2.0\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --difficulty <bits>   Number of matching bits required (default: 50)\n";
            std::cout << "  --preimage <hex>      64-character hex string for 32-byte preimage\n";
            std::cout << "  --help                Show this help message\n\n";
            std::cout << "Example:\n";
            std::cout << "  " << argv[0] << " --difficulty 55 --preimage ";
            std::cout << "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
            return 0;
        }
    }

    // Generate random preimage if not provided
    if (use_random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        std::cout << "Generating random preimage...\n";
        for (int i = 0; i < 32; i++) {
            preimage[i] = dis(gen);
        }
    }

    SHA1CollisionMiner miner(difficulty);
    miner.start_mining(preimage, difficulty);

    return 0;
}