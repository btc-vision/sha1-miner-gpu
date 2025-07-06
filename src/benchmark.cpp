// benchmark.cpp - Performance benchmarking tool
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include "sha1_miner_core.cuh"
#include "cxxsha1.hpp"
#include "job_upload_api.h"

extern "C" __global__ void sha1_collision_mine_kernel(
    CollisionCandidate* __restrict__ candidates,
    uint32_t* __restrict__ candidate_count,
    uint32_t* __restrict__ best_distance,
    const uint64_t total_threads
);

void benchmark_kernel(int device_id, int blocks, int threads, int iterations) {
    cudaSetDevice(device_id);

    // Setup dummy job
    MiningJob job;
    memset(&job, 0, sizeof(job));
    job.difficulty_bits = 20;
    job.nonce_start = 0;
    job.nonce_range = 1ull << 36;

    uint8_t dummy_msg[32] = {0};
    uint32_t dummy_target[5] = {0};

    upload_new_job(dummy_msg, dummy_target);
    cudaMemcpyToSymbol(g_job, &job, sizeof(MiningJob));

    // Initialize bit masks
    uint32_t bit_masks[32];
    for (int i = 0; i < 32; i++) {
        bit_masks[i] = 1u << (31 - i);
    }
    cudaMemcpyToSymbol(g_bit_masks, bit_masks, sizeof(bit_masks));

    // Allocate memory
    CollisionCandidate* d_candidates;
    uint32_t* d_candidate_count;
    uint32_t* d_best_distance;

    cudaMalloc(&d_candidates, sizeof(CollisionCandidate) * CANDIDATES_RING_SIZE);
    cudaMalloc(&d_candidate_count, sizeof(uint32_t));
    cudaMalloc(&d_best_distance, sizeof(uint32_t));

    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_candidate_count, 0, sizeof(uint32_t));
        cudaMemset(d_best_distance, 160, sizeof(uint32_t));

        sha1_collision_mine_kernel<<<blocks, threads>>>(
            d_candidates, d_candidate_count, d_best_distance,
            (uint64_t)blocks * threads
        );
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_candidate_count, 0, sizeof(uint32_t));
        cudaMemset(d_best_distance, 160, sizeof(uint32_t));

        sha1_collision_mine_kernel<<<blocks, threads>>>(
            d_candidates, d_candidate_count, d_best_distance,
            (uint64_t)blocks * threads
        );
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Calculate performance
    uint64_t total_hashes = job.nonce_range * iterations;
    double seconds = duration.count() / 1e6;
    double ghps = total_hashes / seconds / 1e9;

    std::cout << "Configuration: " << blocks << " blocks x " << threads << " threads\n";
    std::cout << "Total time: " << seconds << " seconds\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << ghps << " GH/s\n";
    std::cout << "Time per kernel: " << (seconds * 1000 / iterations) << " ms\n\n";

    // Cleanup
    cudaFree(d_candidates);
    cudaFree(d_candidate_count);
    cudaFree(d_best_distance);
}

int main() {
    std::cout << "SHA-1 Collision Mining Benchmark\n";
    std::cout << "================================\n\n";

    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "GPU " << dev << ": " << prop.name << "\n";
        std::cout << "SMs: " << prop.multiProcessorCount << "\n";
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n\n";

        // Test different configurations
        std::vector<std::pair<int, int>> configs = {
            {prop.multiProcessorCount * 16, 256},
            {prop.multiProcessorCount * 32, 256},
            {prop.multiProcessorCount * 32, 128},
            {prop.multiProcessorCount * 64, 64}
        };

        for (auto [blocks, threads] : configs) {
            benchmark_kernel(dev, blocks, threads, 100);
        }

        std::cout << "----------------------------------------\n\n";
    }

    return 0;
}