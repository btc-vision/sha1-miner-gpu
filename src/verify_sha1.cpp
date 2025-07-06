#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cstring>
#include <chrono>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include "sha1_miner_core.cuh"
#include "cxxsha1.hpp"
#include "job_upload_api.h"

// Kernel declaration
extern "C" __global__ void sha1_collision_mine_kernel(
    CollisionCandidate* __restrict__ candidates,
    uint32_t* __restrict__ candidate_count,
    uint32_t* __restrict__ best_distance,
    const uint64_t total_threads
);

extern "C" __global__ void filter_candidates_kernel(
    const CollisionCandidate* __restrict__ all_candidates,
    CollisionCandidate* __restrict__ filtered_candidates,
    const uint32_t total_candidates,
    uint32_t* __restrict__ filtered_count,
    const uint32_t max_distance_bits
);

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        return false;} \
    }while(0)

#define CUDA_CHECK_VOID(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(1);} \
    }while(0)

// Test vector structure
struct TestVector {
    std::string name;
    std::array<uint8_t, 32> message;
    std::array<uint8_t, 20> expected_hash;
    std::string expected_hash_str;
};

// Test result structure
struct TestResult {
    bool passed;
    bool hash_correct;
    bool found_collision;
    uint32_t distance_found;
    double performance_ghps;
    double kernel_time_ms;
    std::string error_message;
    uint64_t candidates_found;
};

// Generate test vectors using cxxsha1
std::vector<TestVector> generate_test_vectors() {
    std::vector<TestVector> vectors;

    // Test 1: All zeros
    {
        TestVector tv;
        tv.name = "All zeros";
        tv.message.fill(0);

        SHA1 sha1;
        sha1.update(std::string((char*)tv.message.data(), 32));
        tv.expected_hash_str = sha1.final();

        for (int i = 0; i < 20; i++) {
            std::string byte = tv.expected_hash_str.substr(i * 2, 2);
            tv.expected_hash[i] = std::stoul(byte, nullptr, 16);
        }

        vectors.push_back(tv);
    }

    // Test 2: Sequential bytes
    {
        TestVector tv;
        tv.name = "Sequential bytes";
        for (int i = 0; i < 32; i++) tv.message[i] = i;

        SHA1 sha1;
        sha1.update(std::string((char*)tv.message.data(), 32));
        tv.expected_hash_str = sha1.final();

        for (int i = 0; i < 20; i++) {
            std::string byte = tv.expected_hash_str.substr(i * 2, 2);
            tv.expected_hash[i] = std::stoul(byte, nullptr, 16);
        }

        vectors.push_back(tv);
    }

    // Test 3: All 0xFF
    {
        TestVector tv;
        tv.name = "All 0xFF";
        tv.message.fill(0xFF);

        SHA1 sha1;
        sha1.update(std::string((char*)tv.message.data(), 32));
        tv.expected_hash_str = sha1.final();

        for (int i = 0; i < 20; i++) {
            std::string byte = tv.expected_hash_str.substr(i * 2, 2);
            tv.expected_hash[i] = std::stoul(byte, nullptr, 16);
        }

        vectors.push_back(tv);
    }

    // Test 4: Pattern
    {
        TestVector tv;
        tv.name = "Pattern 0xDEADBEEF";
        for (int i = 0; i < 32; i += 4) {
            tv.message[i] = 0xDE;
            tv.message[i + 1] = 0xAD;
            tv.message[i + 2] = 0xBE;
            tv.message[i + 3] = 0xEF;
        }

        SHA1 sha1;
        sha1.update(std::string((char*)tv.message.data(), 32));
        tv.expected_hash_str = sha1.final();

        for (int i = 0; i < 20; i++) {
            std::string byte = tv.expected_hash_str.substr(i * 2, 2);
            tv.expected_hash[i] = std::stoul(byte, nullptr, 16);
        }

        vectors.push_back(tv);
    }

    // Test 5: ASCII text
    {
        TestVector tv;
        tv.name = "ASCII text";
        const char* text = "The quick brown fox jumps over.."; // 32 chars
        memcpy(tv.message.data(), text, 32);

        SHA1 sha1;
        sha1.update(std::string((char*)tv.message.data(), 32));
        tv.expected_hash_str = sha1.final();

        for (int i = 0; i < 20; i++) {
            std::string byte = tv.expected_hash_str.substr(i * 2, 2);
            tv.expected_hash[i] = std::stoul(byte, nullptr, 16);
        }

        vectors.push_back(tv);
    }

    // Test 6: Random (fixed seed for reproducibility)
    {
        TestVector tv;
        tv.name = "Random (seed=12345)";
        std::mt19937 gen(12345);
        std::uniform_int_distribution<> dis(0, 255);

        for (int i = 0; i < 32; i++) {
            tv.message[i] = dis(gen);
        }

        SHA1 sha1;
        sha1.update(std::string((char*)tv.message.data(), 32));
        tv.expected_hash_str = sha1.final();

        for (int i = 0; i < 20; i++) {
            std::string byte = tv.expected_hash_str.substr(i * 2, 2);
            tv.expected_hash[i] = std::stoul(byte, nullptr, 16);
        }

        vectors.push_back(tv);
    }

    return vectors;
}

// Test correctness of the kernel
bool test_kernel_correctness(const TestVector& tv, TestResult& result) {
    // Setup mining job
    MiningJob job;
    memcpy(job.base_msg, tv.message.data(), 32);

    // Convert expected hash to target format
    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(tv.expected_hash[4 * i]) << 24) |
                    (uint32_t(tv.expected_hash[4 * i + 1]) << 16) |
                    (uint32_t(tv.expected_hash[4 * i + 2]) << 8) |
                    uint32_t(tv.expected_hash[4 * i + 3]);
        job.target_hash[i] = target[i];
    }

    job.difficulty_bits = 160; // Looking for exact match
    job.nonce_start = 0;
    job.nonce_range = 1024; // Small range for testing

    // Upload job
    upload_new_job(tv.message.data(), target);
    CUDA_CHECK(cudaMemcpyToSymbol(g_job, &job, sizeof(MiningJob)));

    // Allocate device memory
    CollisionCandidate* d_candidates;
    uint32_t* d_candidate_count;
    uint32_t* d_best_distance;

    CUDA_CHECK(cudaMalloc(&d_candidates, sizeof(CollisionCandidate) * CANDIDATES_RING_SIZE));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_best_distance, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_best_distance, 160, sizeof(uint32_t)));

    // Launch kernel with small configuration
    dim3 grid(32);
    dim3 block(256);
    uint64_t total_threads = grid.x * block.x;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    sha1_collision_mine_kernel<<<grid, block>>>(
        d_candidates, d_candidate_count, d_best_distance, total_threads
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    result.kernel_time_ms = milliseconds;

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_message = cudaGetErrorString(err);
        result.passed = false;
        CUDA_CHECK(cudaFree(d_candidates));
        CUDA_CHECK(cudaFree(d_candidate_count));
        CUDA_CHECK(cudaFree(d_best_distance));
        return false;
    }

    // Get results
    uint32_t found_count = 0;
    uint32_t best_distance = 160;

    CUDA_CHECK(cudaMemcpy(&found_count, d_candidate_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_distance, d_best_distance, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result.candidates_found = found_count;
    result.distance_found = best_distance;

    // The kernel should find that nonce=0 gives the exact hash
    result.found_collision = (best_distance == 0 && found_count > 0);

    if (found_count > 0 && best_distance == 0) {
        // Verify the found collision
        CollisionCandidate h_candidate;
        CUDA_CHECK(cudaMemcpy(&h_candidate, d_candidates, sizeof(CollisionCandidate),
                             cudaMemcpyDeviceToHost));

        // Reconstruct message with nonce
        uint8_t test_msg[32];
        memcpy(test_msg, tv.message.data(), 32);
        uint32_t* msg_words = (uint32_t*)test_msg;
        msg_words[6] ^= (uint32_t)(h_candidate.nonce & 0xFFFFFFFF);
        msg_words[7] ^= (uint32_t)(h_candidate.nonce >> 32);

        // Compute SHA-1
        SHA1 sha1;
        sha1.update(std::string((char*)test_msg, 32));
        std::string computed_hash = sha1.final();

        result.hash_correct = (computed_hash == tv.expected_hash_str);
        result.passed = result.found_collision && result.hash_correct;
    } else {
        result.hash_correct = false;
        result.passed = false;
        if (found_count == 0) {
            result.error_message = "No candidates found";
        } else {
            result.error_message = "Found candidates but no exact match (distance=" +
                                  std::to_string(best_distance) + ")";
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_best_distance));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}

// Test kernel performance
bool test_kernel_performance(TestResult& result) {
    // Setup a dummy job for performance testing
    MiningJob job;
    memset(&job, 0, sizeof(job));
    job.difficulty_bits = 20; // Low difficulty for more candidates
    job.nonce_start = 0;
    job.nonce_range = 1ull << 34; // Large range

    uint8_t dummy_msg[32] = {0};
    uint32_t dummy_target[5] = {0};

    upload_new_job(dummy_msg, dummy_target);
    CUDA_CHECK(cudaMemcpyToSymbol(g_job, &job, sizeof(MiningJob)));

    // Allocate device memory
    CollisionCandidate* d_candidates;
    uint32_t* d_candidate_count;
    uint32_t* d_best_distance;

    CUDA_CHECK(cudaMalloc(&d_candidates, sizeof(CollisionCandidate) * CANDIDATES_RING_SIZE));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_best_distance, sizeof(uint32_t)));

    // Get device properties for optimal configuration
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    dim3 grid(prop.multiProcessorCount * 32);
    dim3 block(256);
    uint64_t total_threads = grid.x * block.x;

    // Warmup
    for (int i = 0; i < 10; i++) {
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_best_distance, 160, sizeof(uint32_t)));

        sha1_collision_mine_kernel<<<grid, block>>>(
            d_candidates, d_candidate_count, d_best_distance, total_threads
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_best_distance, 160, sizeof(uint32_t)));

        sha1_collision_mine_kernel<<<grid, block>>>(
            d_candidates, d_candidate_count, d_best_distance, total_threads
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate performance
    // Each thread processes nonce_range/total_threads nonces
    uint64_t hashes_per_iteration = job.nonce_range;
    uint64_t total_hashes = hashes_per_iteration * iterations;

    result.kernel_time_ms = milliseconds / iterations;
    result.performance_ghps = total_hashes / (milliseconds * 1e6);

    // Check for reasonable performance (should be 50-200 GH/s on modern GPUs)
    if (result.performance_ghps < 10) {
        result.error_message = "Performance too low - possible kernel bug";
    } else if (result.performance_ghps > 500) {
        result.error_message = "Performance suspiciously high - kernel may have early exit bug";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_best_distance));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}

// Test near-collision detection
bool test_near_collision_detection(TestResult& result) {
    // Create a job where we're looking for near-collisions
    MiningJob job;

    // Use a known preimage
    uint8_t preimage[32] = {0};
    memcpy(job.base_msg, preimage, 32);

    // Calculate its hash
    SHA1 sha1;
    sha1.update(std::string((char*)preimage, 32));
    std::string hash_str = sha1.final();

    // Convert to binary
    uint8_t hash_bytes[20];
    for (int i = 0; i < 20; i++) {
        std::string byte = hash_str.substr(i * 2, 2);
        hash_bytes[i] = std::stoul(byte, nullptr, 16);
    }

    // Slightly modify the target to create a near-collision scenario
    // Flip a few bits in the last bytes
    hash_bytes[19] ^= 0x0F;  // Flip 4 bits
    hash_bytes[18] ^= 0x03;  // Flip 2 bits

    // Convert to uint32_t array
    uint32_t target[5];
    for (int i = 0; i < 5; i++) {
        target[i] = (uint32_t(hash_bytes[i*4]) << 24) |
                    (uint32_t(hash_bytes[i*4+1]) << 16) |
                    (uint32_t(hash_bytes[i*4+2]) << 8) |
                    uint32_t(hash_bytes[i*4+3]);
        job.target_hash[i] = target[i];
    }

    job.difficulty_bits = 154; // We flipped 6 bits, so we want 154+ matching bits
    job.nonce_start = 0;
    job.nonce_range = 1ull << 20; // 1M nonces

    // Upload job
    upload_new_job(preimage, target);
    CUDA_CHECK(cudaMemcpyToSymbol(g_job, &job, sizeof(MiningJob)));

    // Allocate device memory
    CollisionCandidate* d_candidates;
    uint32_t* d_candidate_count;
    uint32_t* d_best_distance;

    CUDA_CHECK(cudaMalloc(&d_candidates, sizeof(CollisionCandidate) * CANDIDATES_RING_SIZE));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_best_distance, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_best_distance, 160, sizeof(uint32_t)));

    // Launch kernel
    dim3 grid(64);
    dim3 block(256);
    uint64_t total_threads = grid.x * block.x;

    sha1_collision_mine_kernel<<<grid, block>>>(
        d_candidates, d_candidate_count, d_best_distance, total_threads
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Get results
    uint32_t found_count = 0;
    uint32_t best_distance = 160;

    CUDA_CHECK(cudaMemcpy(&found_count, d_candidate_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_distance, d_best_distance, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result.candidates_found = found_count;
    result.distance_found = best_distance;

    // We should find at least one candidate with distance 6
    result.passed = (found_count > 0 && best_distance <= 6);

    if (!result.passed) {
        result.error_message = "Failed to find near-collision (found=" +
                              std::to_string(found_count) + ", best_distance=" +
                              std::to_string(best_distance) + ")";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_best_distance));

    return true;
}

// Main test function
void run_all_tests() {
    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|   SHA-1 Collision Kernel Test Suite     |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Get device info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK_VOID(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB\n\n";

    // Initialize bit masks
    uint32_t bit_masks[32];
    for (int i = 0; i < 32; i++) {
        bit_masks[i] = 1u << (31 - i);
    }
    CUDA_CHECK_VOID(cudaMemcpyToSymbol(g_bit_masks, bit_masks, sizeof(bit_masks)));

    // Generate test vectors
    auto test_vectors = generate_test_vectors();

    std::cout << "=== Correctness Tests ===\n";
    int passed = 0, failed = 0;

    for (const auto& tv : test_vectors) {
        std::cout << "Testing: " << std::setw(25) << std::left << tv.name << " ... ";

        TestResult result;
        if (!test_kernel_correctness(tv, result)) {
            std::cout << "ERROR (kernel failed)\n";
            failed++;
            continue;
        }

        if (result.passed) {
            std::cout << "PASS (" << std::fixed << std::setprecision(2)
                     << result.kernel_time_ms << " ms)\n";
            passed++;
        } else {
            std::cout << "FAIL (" << result.error_message << ")\n";
            failed++;
        }
    }

    std::cout << "\n=== Near-Collision Detection Test ===\n";
    {
        std::cout << "Testing near-collision detection ... ";
        TestResult result;
        if (test_near_collision_detection(result)) {
            if (result.passed) {
                std::cout << "PASS (found " << result.candidates_found
                         << " candidates, best distance: " << result.distance_found << " bits)\n";
                passed++;
            } else {
                std::cout << "FAIL (" << result.error_message << ")\n";
                failed++;
            }
        } else {
            std::cout << "ERROR\n";
            failed++;
        }
    }

    std::cout << "\n=== Performance Test ===\n";
    {
        TestResult result;
        if (test_kernel_performance(result)) {
            std::cout << "Hash rate: " << std::fixed << std::setprecision(2)
                     << result.performance_ghps << " GH/s\n";
            std::cout << "Kernel time: " << result.kernel_time_ms << " ms/iteration\n";

            if (!result.error_message.empty()) {
                std::cout << "WARNING: " << result.error_message << "\n";
            }
        } else {
            std::cout << "Performance test failed\n";
        }
    }

    // Summary
    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|              TEST SUMMARY                |\n";
    std::cout << "+------------------------------------------+\n";
    std::cout << "Total tests: " << (passed + failed) << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    if (failed == 0) {
        std::cout << "\nAll tests PASSED! Kernel is working correctly.\n";
    } else {
        std::cout << "\nSome tests FAILED. Please check the kernel implementation.\n";
    }

    // Write report
    std::ofstream report("kernel_verification_report.txt");
    if (report.is_open()) {
        report << "SHA-1 Collision Kernel Verification Report\n";
        report << "==========================================\n\n";
        report << "GPU: " << prop.name << "\n";
        report << "Date: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n\n";
        report << "Test Results:\n";
        report << "  Correctness tests: " << passed << "/" << test_vectors.size() << " passed\n";
        report << "  Near-collision test: " << (failed == 0 ? "PASSED" : "FAILED") << "\n";
        report << "\nStatus: " << (failed == 0 ? "VERIFIED" : "NEEDS ATTENTION") << "\n";
        report.close();

        std::cout << "\nDetailed report written to: kernel_verification_report.txt\n";
    }
}

int main(int argc, char** argv) {
    run_all_tests();

    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}