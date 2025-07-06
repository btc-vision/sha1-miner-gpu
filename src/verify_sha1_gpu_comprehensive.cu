#include "sha1_miner.cuh"
#include "mining_system.hpp"
#include "utilities.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <algorithm>

// Forward declarations
extern "C" bool test_gpu_sha1_implementation();

extern "C" void launch_mining_kernel(
    const MiningJob &job,
    const ResultPool &pool,
    const KernelConfig &config
);

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string details;
    double time_ms;
};

// Reference CPU implementation for verification
void compute_sha1_with_nonce_cpu_ref(const uint8_t *message, uint64_t nonce, uint8_t *hash_out) {
    std::vector<uint8_t> msg_copy(message, message + 32);

    // Apply nonce the same way GPU does - XOR with last 8 bytes in big-endian
    for (int i = 0; i < 8; i++) {
        msg_copy[24 + i] ^= (nonce >> (56 - i * 8)) & 0xFF;
    }

    auto hash = calculate_sha1(msg_copy);
    std::memcpy(hash_out, hash.data(), 20);
}

// Test 1: Boundary nonce values
TestResult test_boundary_nonces() {
    TestResult result{"Boundary Nonce Values", true, "", 0.0};
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Test 1: Boundary Nonce Values ===\n";
    std::cout << "Testing nonces at boundaries to ensure no overflow issues...\n";

    // Test message
    std::vector<uint8_t> message(32);
    std::strcpy(reinterpret_cast<char *>(message.data()), "Boundary test message");

    // Boundary nonces to test
    std::vector<uint64_t> boundary_nonces = {
        0, 1, 0xFF, 0xFFFF, 0xFFFFFFFF,
        0xFFFFFFFFFFFFFFFFULL - 1, 0xFFFFFFFFFFFFFFFFULL,
        0x8000000000000000ULL, 0x0123456789ABCDEFULL
    };

    // Create stream for testing
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int errors = 0;
    for (uint64_t test_nonce: boundary_nonces) {
        // Create job with a known target
        MiningJob job;
        std::memcpy(job.base_message, message.data(), 32);
        job.nonce_offset = test_nonce;
        job.difficulty = 8; // Low difficulty

        // Create a dummy target
        std::memset(job.target_hash, 0xFF, sizeof(job.target_hash));

        // Prepare result pool
        ResultPool pool;
        pool.capacity = 10;
        cudaMalloc(&pool.results, sizeof(MiningResult) * pool.capacity);
        cudaMalloc(&pool.count, sizeof(uint32_t));
        cudaMemset(pool.count, 0, sizeof(uint32_t));

        // Launch kernel with minimal configuration
        KernelConfig config;
        config.blocks = 1;
        config.threads_per_block = 32;
        config.shared_memory_size = 0;
        config.stream = stream;

        launch_mining_kernel(job, pool, config);

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "  Nonce 0x" << std::hex << test_nonce
                    << ": Kernel launch failed - " << cudaGetErrorString(err) << "\n";
            errors++;
        } else {
            cudaStreamSynchronize(stream);

            uint32_t count;
            cudaMemcpy(&count, pool.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            std::cout << "  Nonce 0x" << std::hex << test_nonce << ": ";
            if (count > pool.capacity) {
                std::cout << "OVERFLOW! Count=" << std::dec << count << "\n";
                errors++;
            } else {
                std::cout << "OK (found " << std::dec << count << " results)\n";
            }
        }

        // Cleanup
        cudaFree(pool.results);
        cudaFree(pool.count);
    }

    cudaStreamDestroy(stream);

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (errors > 0) {
        result.passed = false;
        result.details = "Found " + std::to_string(errors) + " errors";
    } else {
        result.details = "All boundary nonces handled correctly";
    }

    return result;
}

// Test 2: Result accuracy verification
TestResult test_result_accuracy() {
    TestResult result{"Result Accuracy Verification", true, "", 0.0};
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Test 2: Result Accuracy Verification ===\n";
    std::cout << "Verifying that GPU computes correct SHA-1 hashes...\n";

    // Create a test message
    std::vector<uint8_t> message(32);
    for (int i = 0; i < 32; i++) {
        message[i] = i * 7 + 3;
    }

    // Create job
    MiningJob job;
    std::memcpy(job.base_message, message.data(), 32);
    std::memset(job.target_hash, 0xFF, sizeof(job.target_hash));
    job.difficulty = 16; // Low difficulty to get results
    job.nonce_offset = 1000000; // Start from non-zero

    // Prepare result pool
    ResultPool pool;
    pool.capacity = 100;
    cudaMalloc(&pool.results, sizeof(MiningResult) * pool.capacity);
    cudaMalloc(&pool.count, sizeof(uint32_t));
    cudaMemset(pool.count, 0, sizeof(uint32_t));

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch kernel
    KernelConfig config;
    config.blocks = 8;
    config.threads_per_block = 128;
    config.shared_memory_size = 0;
    config.stream = stream;

    launch_mining_kernel(job, pool, config);
    cudaStreamSynchronize(stream);

    // Get results
    uint32_t count;
    cudaMemcpy(&count, pool.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    count = std::min(count, pool.capacity);

    std::cout << "  Found " << count << " candidates to verify\n";

    int errors = 0;
    if (count > 0) {
        std::vector<MiningResult> results(count);
        cudaMemcpy(results.data(), pool.results, sizeof(MiningResult) * count, cudaMemcpyDeviceToHost);

        // Verify each result
        for (size_t i = 0; i < std::min(count, 10u); i++) {
            const auto &res = results[i];

            // Compute expected hash
            uint8_t cpu_hash[20];
            compute_sha1_with_nonce_cpu_ref(message.data(), res.nonce, cpu_hash);

            // Compare
            bool match = true;
            for (int j = 0; j < 5; j++) {
                uint32_t cpu_word = (cpu_hash[j * 4] << 24) | (cpu_hash[j * 4 + 1] << 16) |
                                    (cpu_hash[j * 4 + 2] << 8) | cpu_hash[j * 4 + 3];
                if (res.hash[j] != cpu_word) {
                    match = false;
                    break;
                }
            }

            if (!match) {
                errors++;
                std::cout << "  ERROR: Hash mismatch for nonce " << res.nonce << "\n";
            } else if (i < 3) {
                // Show first few results
                std::cout << "  Verified nonce " << res.nonce << " - "
                        << res.matching_bits << " bits match\n";
            }
        }

        if (count > 10) {
            std::cout << "  (verified first 10 results)\n";
        }
    }

    // Cleanup
    cudaFree(pool.results);
    cudaFree(pool.count);
    cudaStreamDestroy(stream);

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (errors > 0) {
        result.passed = false;
        result.details = "Found " + std::to_string(errors) + " hash mismatches";
    } else {
        result.details = "All hashes verified correctly";
    }

    return result;
}

// Test 3: Thread configuration stress test
TestResult test_thread_configurations() {
    TestResult result{"Thread Configuration Stress Test", true, "", 0.0};
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Test 3: Thread Configuration Stress Test ===\n";
    std::cout << "Testing various block and thread configurations...\n";

    std::vector<uint8_t> message(32, 0xAA);

    // Test configurations
    struct Config {
        int blocks;
        int threads;
        const char *name;
    };

    std::vector<Config> configs = {
        {1, 32, "Minimal (1x32)"},
        {1, 256, "Single block (1x256)"},
        {32, 128, "Small (32x128)"},
        {64, 256, "Medium (64x256)"},
        {256, 256, "Large (256x256)"},
        {1, 1024, "Max threads (1x1024)"}
    };

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (const auto &cfg: configs) {
        std::cout << "  Testing " << cfg.name << "... ";

        MiningJob job;
        std::memcpy(job.base_message, message.data(), 32);
        std::memset(job.target_hash, 0, sizeof(job.target_hash));
        job.difficulty = 20;
        job.nonce_offset = 0;

        ResultPool pool;
        pool.capacity = 100;
        cudaMalloc(&pool.results, sizeof(MiningResult) * pool.capacity);
        cudaMalloc(&pool.count, sizeof(uint32_t));
        cudaMemset(pool.count, 0, sizeof(uint32_t));

        KernelConfig config;
        config.blocks = cfg.blocks;
        config.threads_per_block = cfg.threads;
        config.shared_memory_size = 0;
        config.stream = stream;

        launch_mining_kernel(job, pool, config);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "FAILED: " << cudaGetErrorString(err) << "\n";
            result.passed = false;
            result.details += std::string(cfg.name) + " failed. ";
        } else {
            cudaStreamSynchronize(stream);

            uint32_t count;
            cudaMemcpy(&count, pool.count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (count > pool.capacity) {
                std::cout << "OVERFLOW (count=" << count << ")\n";
                result.passed = false;
            } else {
                std::cout << "OK\n";
            }
        }

        cudaFree(pool.results);
        cudaFree(pool.count);
    }

    cudaStreamDestroy(stream);

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (result.passed) {
        result.details = "All configurations handled correctly";
    }

    return result;
}

// Test 4: Performance measurement
TestResult test_performance_measurement() {
    TestResult result{"Performance Measurement", true, "", 0.0};
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Test 4: Performance Measurement ===\n";
    std::cout << "Measuring hash rate with optimal configuration...\n";

    // Get device properties
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    std::vector<uint8_t> message(32);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<> dist(0, 255);
    for (auto &b: message) {
        b = dist(rng);
    }

    MiningJob job;
    std::memcpy(job.base_message, message.data(), 32);
    for (int i = 0; i < 5; i++) {
        job.target_hash[i] = dist(rng) << 24 | dist(rng) << 16 | dist(rng) << 8 | dist(rng);
    }
    job.difficulty = 100; // High difficulty
    job.nonce_offset = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ResultPool pool;
    pool.capacity = 1000;
    cudaMalloc(&pool.results, sizeof(MiningResult) * pool.capacity);
    cudaMalloc(&pool.count, sizeof(uint32_t));

    // Calculate optimal configuration
    int blocks = props.multiProcessorCount * 2;
    int threads = 256;

    KernelConfig config;
    config.blocks = blocks;
    config.threads_per_block = threads;
    config.shared_memory_size = 0;
    config.stream = stream;

    // Warm up
    cudaMemset(pool.count, 0, sizeof(uint32_t));
    launch_mining_kernel(job, pool, config);
    cudaStreamSynchronize(stream);

    // Timed run
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    const int num_iterations = 10;
    const uint64_t hashes_per_iteration = static_cast<uint64_t>(blocks) * threads * NONCES_PER_THREAD;

    cudaEventRecord(start_event, stream);
    for (int i = 0; i < num_iterations; i++) {
        cudaMemset(pool.count, 0, sizeof(uint32_t));
        job.nonce_offset = i * hashes_per_iteration;
        launch_mining_kernel(job, pool, config);
    }
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    double total_hashes = static_cast<double>(hashes_per_iteration) * num_iterations;
    double hash_rate = total_hashes / (milliseconds / 1000.0) / 1e9; // GH/s

    std::cout << "  Configuration: " << blocks << " blocks x " << threads << " threads\n";
    std::cout << "  Total hashes: " << std::scientific << total_hashes << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << milliseconds << " ms\n";
    std::cout << "  Hash rate: " << std::fixed << std::setprecision(2) << hash_rate << " GH/s\n";

    // Cleanup
    cudaFree(pool.results);
    cudaFree(pool.count);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.details = "Hash rate: " + std::to_string(hash_rate) + " GH/s";

    return result;
}

// Main comprehensive test function
extern "C" void run_comprehensive_gpu_tests() {
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "   Comprehensive GPU Kernel Verification\n";
    std::cout << "==============================================\n";

    // First run basic SHA-1 test
    std::cout << "\nRunning basic SHA-1 implementation test...\n";
    bool basic_test_passed = test_gpu_sha1_implementation();
    if (!basic_test_passed) {
        std::cout << "\nERROR: Basic SHA-1 test failed! Aborting further tests.\n";
        return;
    }

    std::vector<TestResult> results;

    // Run all tests
    results.push_back(test_boundary_nonces());
    results.push_back(test_result_accuracy());
    results.push_back(test_thread_configurations());
    results.push_back(test_performance_measurement());

    // Summary
    std::cout << "\n==============================================\n";
    std::cout << "                Test Summary\n";
    std::cout << "==============================================\n";

    int passed = 0;
    double total_time = 0;

    for (const auto &result: results) {
        std::cout << std::left << std::setw(35) << result.test_name << ": ";
        std::cout << (result.passed ? "PASS" : "FAIL");
        std::cout << " (" << std::fixed << std::setprecision(1) << result.time_ms << " ms)\n";

        if (!result.details.empty()) {
            std::cout << "    " << result.details << "\n";
        }

        if (result.passed) passed++;
        total_time += result.time_ms;
    }

    std::cout << "\nTotal: " << passed << "/" << results.size() << " tests passed\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_time << " ms\n";

    if (passed != results.size()) {
        std::cout << "\nWARNING: Some tests failed! The kernel may have issues.\n";
    } else {
        std::cout << "\nAll tests passed! The kernel appears to be working correctly.\n";
    }
}
