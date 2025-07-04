// test_shattered.cpp - Test program for SHAttered-style collision kernel with hashrate logging

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include "cxxsha1.hpp"
#include "job_upload_api.h"

// Declare the new kernel functions
extern "C" __global__ void sha1_shattered_collision_kernel(uint64_t *, uint32_t *, uint64_t);

extern "C" __global__ void sha1_shattered_debug_kernel(uint32_t *, uint64_t);

extern "C" __global__ void init_shattered_constants();

#define CUDA_CHECK(e) do{ cudaError_t _e=(e); \
    if(_e!=cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        return false;} \
    }while(0)

// Performance tracking
struct PerformanceMetrics {
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<uint64_t> total_iterations{0};
    std::chrono::high_resolution_clock::time_point start_time;
    std::mutex print_mutex;

    void reset() {
        total_hashes = 0;
        total_iterations = 0;
        start_time = std::chrono::high_resolution_clock::now();
    }

    void update(uint64_t hashes, uint64_t iterations) {
        total_hashes += hashes;
        total_iterations += iterations;
    }

    void print_stats() {
        std::lock_guard<std::mutex> lock(print_mutex);
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        if (duration > 0) {
            double seconds = duration / 1000.0;
            double hash_rate = total_hashes.load() / seconds;
            double mega_hash_rate = hash_rate / 1e6;
            double giga_hash_rate = hash_rate / 1e9;

            std::cout << "\r[Performance] "
                    << "Time: " << std::fixed << std::setprecision(1) << seconds << "s | "
                    << "Iterations: " << total_iterations.load() << " | "
                    << "Total Hashes: " << total_hashes.load() << " | ";

            if (giga_hash_rate >= 1.0) {
                std::cout << "Rate: " << std::setprecision(2) << giga_hash_rate << " GH/s";
            } else {
                std::cout << "Rate: " << std::setprecision(2) << mega_hash_rate << " MH/s";
            }

            std::cout << " | Avg: " << std::setprecision(0)
                    << (total_hashes.load() / (total_iterations.load() + 1)) << " H/iter";

            std::cout << std::flush;
        }
    }

    void print_final() {
        std::cout << std::endl; // New line after progress updates
        print_stats();
        std::cout << std::endl;

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        double seconds = duration / 1000.0;

        std::cout << "\n=== Final Performance Summary ===" << std::endl;
        std::cout << "Total Runtime: " << std::fixed << std::setprecision(2) << seconds << " seconds" << std::endl;
        std::cout << "Total Iterations: " << total_iterations.load() << std::endl;
        std::cout << "Total Hash Operations: " << std::scientific << std::setprecision(3)
                << (double) total_hashes.load() << std::endl;

        if (seconds > 0) {
            double avg_rate = total_hashes.load() / seconds;
            std::cout << "Average Hash Rate: " << std::fixed << std::setprecision(2);
            if (avg_rate >= 1e9) {
                std::cout << avg_rate / 1e9 << " GH/s" << std::endl;
            } else if (avg_rate >= 1e6) {
                std::cout << avg_rate / 1e6 << " MH/s" << std::endl;
            } else {
                std::cout << avg_rate / 1e3 << " KH/s" << std::endl;
            }
        }

        // Theoretical time estimates
        std::cout << "\n=== Theoretical Time Estimates ===" << std::endl;
        if (total_hashes.load() > 0 && seconds > 0) {
            double rate = total_hashes.load() / seconds;

            // Time for 2^63 operations (SHAttered complexity)
            double ops_2_63 = pow(2.0, 63);
            double seconds_2_63 = ops_2_63 / rate;
            double days_2_63 = seconds_2_63 / (24 * 3600);
            double years_2_63 = days_2_63 / 365.25;

            std::cout << "Time for 2^63 operations (SHAttered): ";
            if (years_2_63 > 1) {
                std::cout << std::fixed << std::setprecision(1) << years_2_63 << " years" << std::endl;
            } else if (days_2_63 > 1) {
                std::cout << std::fixed << std::setprecision(1) << days_2_63 << " days" << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(1) << seconds_2_63 / 3600 << " hours" << std::endl;
            }

            // Time for 2^80 operations (brute force)
            double ops_2_80 = pow(2.0, 80);
            double years_2_80 = (ops_2_80 / rate) / (365.25 * 24 * 3600);
            std::cout << "Time for 2^80 operations (brute force): "
                    << std::scientific << std::setprecision(2) << years_2_80 << " years" << std::endl;
        }
    }
};

// Test collision detection
bool verify_collision(const uint8_t *msg1, const uint8_t *msg2, size_t len) {
    uint8_t hash1[20], hash2[20];

    sha1_ctx ctx1, ctx2;
    sha1_init(ctx1);
    sha1_init(ctx2);

    sha1_update(ctx1, msg1, len);
    sha1_update(ctx2, msg2, len);

    sha1_final(ctx1, hash1);
    sha1_final(ctx2, hash2);

    return memcmp(hash1, hash2, 20) == 0 && memcmp(msg1, msg2, len) != 0;
}

// Test the SHAttered kernel
bool test_shattered_kernel() {
    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|   SHAttered-Style Collision Attack Test  |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Get device properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n\n";

    // Initialize constants
    std::cout << "Initializing differential paths and near-collision blocks...\n";
    init_shattered_constants<<<1, 64>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate memory
    uint64_t *d_pairs;
    uint32_t *d_ticket;
    uint32_t *d_debug;

    // Note: SHAttered kernel outputs 2 blocks (128 bytes) per collision
    size_t collision_size = sizeof(uint64_t) * 16; // 2 blocks of 64 bytes each
    CUDA_CHECK(cudaMalloc(&d_pairs, collision_size * (1u << 20)));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_debug, sizeof(uint32_t) * 16));

    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_debug, 0, sizeof(uint32_t) * 16));

    // Test 1: Debug kernel
    std::cout << "\nTest 1: Running debug kernel...\n";
    sha1_shattered_debug_kernel<<<1, 1>>>(d_debug, 0x1234567890ABCDEF);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_debug[16];
    CUDA_CHECK(cudaMemcpy(h_debug, d_debug, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost));

    if (h_debug[0] == 0xDEADBEEF) {
        std::cout << "Debug kernel executed successfully\n";
        std::cout << "Test hash: ";
        for (int i = 1; i < 6; i++) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << h_debug[i] << " ";
        }
        std::cout << std::dec << "\n";
    } else {
        std::cout << "Debug kernel failed!\n";
        return false;
    }

    // Test 2: Collision finding (simplified test)
    std::cout << "\nTest 2: Attempting collision attack with performance monitoring...\n";

    // Set up a job with a specific target
    uint8_t prefix[32] = {0};
    memcpy(prefix, "SHA1 is broken!!", 16);

    uint32_t target[5];
    // Use a target that's more likely to have collisions
    target[0] = 0x00000000; // Leading zeros make collisions easier to find
    target[1] = 0x11111111;
    target[2] = 0x22222222;
    target[3] = 0x33333333;
    target[4] = 0x44444444;

    upload_new_job(prefix, target);
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(uint32_t)));

    // Launch configuration for SHAttered kernel
    int blocks = prop.multiProcessorCount * 4;
    int threads = 256;

    // Calculate hashes per kernel launch
    // From the kernel: TRIALS_PER_THREAD = 256, and each trial tests multiple neutral bits
    const uint64_t TRIALS_PER_THREAD = 256;
    const uint64_t NEUTRAL_BITS_TESTED = 16; // nb2 loop in kernel
    const uint64_t hashes_per_thread = TRIALS_PER_THREAD * NEUTRAL_BITS_TESTED;
    const uint64_t hashes_per_launch = (uint64_t) blocks * threads * hashes_per_thread;

    std::cout << "\nKernel Configuration:\n";
    std::cout << "- Blocks: " << blocks << "\n";
    std::cout << "- Threads per block: " << threads << "\n";
    std::cout << "- Total threads: " << blocks * threads << "\n";
    std::cout << "- Trials per thread: " << TRIALS_PER_THREAD << "\n";
    std::cout << "- Neutral bits tested: " << NEUTRAL_BITS_TESTED << "\n";
    std::cout << "- Hashes per thread: " << hashes_per_thread << "\n";
    std::cout << "- Hashes per launch: " << std::scientific << std::setprecision(3)
            << (double) hashes_per_launch << std::fixed << "\n";
    std::cout << "\nThis implements differential cryptanalysis, not brute force\n";
    std::cout << "Expected complexity: ~2^63 instead of 2^80\n\n";

    // Performance tracking
    PerformanceMetrics metrics;
    metrics.reset();

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Time the attack
    CUDA_CHECK(cudaEventRecord(start));

    // Performance monitoring thread
    std::atomic<bool> monitoring{true};
    std::thread monitor_thread([&metrics, &monitoring]() {
        while (monitoring) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (monitoring) {
                metrics.print_stats();
            }
        }
    });

    // Run multiple iterations with different seeds
    const int iterations = 100;
    bool collision_found = false;

    std::cout << "Starting collision search...\n\n";

    for (int i = 0; i < iterations; i++) {
        uint64_t seed = i * 0x123456789ABCDEF;

        // Launch kernel
        sha1_shattered_collision_kernel<<<blocks, threads>>>(d_pairs, d_ticket, seed);

        // Update metrics
        metrics.update(hashes_per_launch, 1);

        // Check if collision found
        uint32_t found;
        CUDA_CHECK(cudaMemcpy(&found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (found > 0) {
            collision_found = true;
            std::cout << "\nPotential collision found at iteration " << i << "!\n";
            break;
        }
    }

    // Stop monitoring
    monitoring = false;
    monitor_thread.join();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Print final metrics
    metrics.print_final();

    std::cout << "\nGPU Timing: " << milliseconds << " ms\n";
    std::cout << "Theoretical GPU Hash Rate: " << std::fixed << std::setprecision(2)
            << (metrics.total_hashes.load() / (milliseconds / 1000.0)) / 1e6 << " MH/s\n";

    // Check results
    uint32_t collisions_found;
    CUDA_CHECK(cudaMemcpy(&collisions_found, d_ticket, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (collisions_found > 0) {
        std::cout << "\nFound " << collisions_found << " collision candidate(s)\n";

        // Verify first collision
        uint64_t h_collision[16];
        CUDA_CHECK(cudaMemcpy(h_collision, d_pairs, collision_size, cudaMemcpyDeviceToHost));

        // Extract the two messages
        uint8_t msg1[64], msg2[64];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                msg1[i * 8 + j] = (h_collision[i] >> (j * 8)) & 0xFF;
                msg2[i * 8 + j] = (h_collision[8 + i] >> (j * 8)) & 0xFF;
            }
        }

        std::cout << "\nMessage 1 (first 32 bytes): ";
        for (int i = 0; i < 32; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int) msg1[i];
        }
        std::cout << "\n\nMessage 2 (first 32 bytes): ";
        for (int i = 0; i < 32; i++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int) msg2[i];
        }
        std::cout << std::dec << "\n\n";

        if (verify_collision(msg1, msg2, 64)) {
            std::cout << "✓ COLLISION VERIFIED! Both messages have the same SHA-1 hash!\n";
        } else {
            std::cout << "✗ Collision verification failed\n";
        }
    } else {
        std::cout << "\nNo collisions found in this test run\n";
        std::cout << "Note: Real SHAttered attack requires ~2^63 operations\n";

        // Calculate progress towards 2^63
        double progress = (double) metrics.total_hashes.load() / pow(2.0, 63);
        std::cout << "Progress towards 2^63: " << std::scientific << std::setprecision(3)
                << progress * 100 << "%\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_ticket));
    CUDA_CHECK(cudaFree(d_debug));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}

// Theoretical analysis
void print_theoretical_analysis() {
    std::cout << "\n+------------------------------------------+\n";
    std::cout << "|        Theoretical Analysis              |\n";
    std::cout << "+------------------------------------------+\n\n";

    std::cout << "SHAttered Attack Complexity:\n";
    std::cout << "- Chosen-prefix collision: ~2^63.1 SHA-1 computations\n";
    std::cout << "- Identical-prefix collision: ~2^61 SHA-1 computations\n";
    std::cout << "- Memory requirement: ~100 MB for differential paths\n\n";

    std::cout << "Key Techniques:\n";
    std::cout << "1. Differential cryptanalysis\n";
    std::cout << "2. Near-collision blocks\n";
    std::cout << "3. Neutral bit manipulation\n";
    std::cout << "4. Boomerang attacks\n";
    std::cout << "5. Birthday paradox optimization\n\n";

    std::cout << "GPU Advantages:\n";
    std::cout << "- Parallel neutral bit search\n";
    std::cout << "- Fast message modification\n";
    std::cout << "- Efficient path condition checking\n";
    std::cout << "- Collaborative intermediate state sharing\n\n";

    std::cout << "Practical Considerations:\n";
    std::cout << "- Full attack requires months on GPUs\n";
    std::cout << "- This kernel demonstrates core concepts\n";
    std::cout << "- Real implementation needs offline precomputation\n";
    std::cout << "- Differential paths must be carefully chosen\n";
}

int main(int argc, char **argv) {
    std::cout << "SHA-1 SHAttered-Style Collision Attack Demo\n";
    std::cout << "==========================================\n\n";

    // Run the test
    if (!test_shattered_kernel()) {
        std::cerr << "Test failed!\n";
        return 1;
    }

    // Print theoretical analysis
    print_theoretical_analysis();

    std::cout << "\nConclusion:\n";
    std::cout << "This kernel demonstrates how cryptanalytic attacks can be\n";
    std::cout << "implemented on GPUs, reducing SHA-1 collision complexity\n";
    std::cout << "from 2^80 (brute force) to 2^63 (differential attack).\n\n";

    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
