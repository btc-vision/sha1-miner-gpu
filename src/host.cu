// host.cu - SHA-1 Collision Attack Host Code
// Fixed implementation with proper parameters

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdlib>

// Performance parameters (must match kernel.cu)
#define BLOCKS_PER_SM 4
#define THREADS_PER_BLOCK 128
#define NEAR_COLLISION_BLOCKS_NEEDED 8192
#define WORK_PER_THREAD 32

// Collision block structure (must match kernel.cu)
struct CollisionBlock {
    uint32_t msg1[16];
    uint32_t msg2[16];
    float quality;
    uint32_t iterations;
};

// External kernel declarations
extern "C" __global__ void find_near_collision_blocks(
    uint64_t base_counter,
    CollisionBlock *output_blocks,
    uint32_t *block_count,
    uint32_t max_blocks,
    uint32_t *global_best_quality
);

extern "C" __global__ void birthday_attack(
    CollisionBlock *blocks,
    uint32_t num_blocks,
    uint32_t *collision_msg1,
    uint32_t *collision_msg2,
    uint32_t *found_flag
);

extern "C" __global__ void collision_attack_status(
    uint64_t total_attempts,
    uint32_t blocks_found,
    uint32_t best_quality_raw,
    uint32_t phase
);

// ==================== Helper Functions ====================

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "\n=== CUDA Device Information ===" << std::endl;

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max blocks per SM: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Total global memory: " << (deviceProp.totalGlobalMem / (1024 * 1024 * 1024)) << " GB" <<
                std::endl;
        std::cout << "  Shared memory per block: " << (deviceProp.sharedMemPerBlock / 1024) << " KB" << std::endl;
    }
}

// ==================== Main Attack Function ====================

int main(int argc, char **argv) {
    std::cout << "\n+------------------------------------------+" << std::endl;
    std::cout << "|    SHA-1 Collision Attack on GPU v1.0    |" << std::endl;
    std::cout << "|         Based on Marc Stevens' work      |" << std::endl;
    std::cout << "+------------------------------------------+" << std::endl;

    // Parse command line arguments
    int device_id = 0;
    int target_blocks = NEAR_COLLISION_BLOCKS_NEEDED;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            target_blocks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            std::cout << "\nUsage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -d <device>  GPU device ID (default: 0)" << std::endl;
            std::cout << "  -b <blocks>  Target near-collision blocks (default: " << NEAR_COLLISION_BLOCKS_NEEDED << ")"
                    << std::endl;
            std::cout << "  -v           Verbose output" << std::endl;
            std::cout << "  -h           Show this help" << std::endl;
            return 0;
        }
    }

    // Initialize CUDA
    cudaSetDevice(device_id);
    checkCudaError(cudaGetLastError(), "Setting device");

    if (verbose) {
        printDeviceInfo();
    }

    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    int num_sms = deviceProp.multiProcessorCount;
    int max_blocks = num_sms * BLOCKS_PER_SM;

    std::cout << "\nUsing device " << device_id << ": " << deviceProp.name << std::endl;
    std::cout << "Launching " << max_blocks << " blocks with " << THREADS_PER_BLOCK << " threads each" << std::endl;
    std::cout << "Target quality threshold: 15%" << std::endl;
    std::cout << "Work per thread: " << WORK_PER_THREAD << " attempts" << std::endl;

    // ==================== Phase 1: Find Near-Collision Blocks ====================

    std::cout << "\n=== Phase 1: Finding Near-Collision Blocks ===" << std::endl;
    std::cout << "Target: " << target_blocks << " blocks" << std::endl;

    // Allocate device memory
    CollisionBlock *d_blocks;
    uint32_t *d_block_count;
    uint32_t *d_best_quality;

    size_t blocks_size = target_blocks * sizeof(CollisionBlock);
    checkCudaError(cudaMalloc(&d_blocks, blocks_size), "Allocating blocks");
    checkCudaError(cudaMalloc(&d_block_count, sizeof(uint32_t)), "Allocating block count");
    checkCudaError(cudaMalloc(&d_best_quality, sizeof(uint32_t)), "Allocating best quality");

    checkCudaError(cudaMemset(d_block_count, 0, sizeof(uint32_t)), "Initializing block count");
    checkCudaError(cudaMemset(d_best_quality, 0, sizeof(uint32_t)), "Initializing best quality");

    // Attack parameters
    uint64_t total_attempts = 0;
    uint32_t blocks_found = 0;
    uint64_t batch_size = (uint64_t) max_blocks * THREADS_PER_BLOCK * WORK_PER_THREAD;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_status = start_time;

    std::cout << "Batch size per kernel launch: " << batch_size << " attempts" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Main search loop
    while (blocks_found < target_blocks) {
        // Launch kernel
        find_near_collision_blocks<<<max_blocks, THREADS_PER_BLOCK>>>(
            total_attempts,
            d_blocks,
            d_block_count,
            target_blocks,
            d_best_quality
        );

        checkCudaError(cudaGetLastError(), "Launching find_near_collision_blocks");

        // Update counters
        total_attempts += batch_size;

        // Check progress periodically
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_since_status = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_status).count();

        if (time_since_status >= 5) {
            // Status every 5 seconds
            checkCudaError(cudaMemcpy(&blocks_found, d_block_count, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                           "Reading block count");

            uint32_t best_quality_raw;
            checkCudaError(cudaMemcpy(&best_quality_raw, d_best_quality, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                           "Reading best quality");

            // Print status
            collision_attack_status<<<1, 1>>>(total_attempts, blocks_found, best_quality_raw, 0);
            checkCudaError(cudaDeviceSynchronize(), "Status kernel");

            last_status = current_time;

            // Calculate actual performance
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            if (elapsed > 0) {
                double hash_rate = (double) total_attempts / (elapsed / 1000.0);
                std::cout << "Actual hash rate: " << std::fixed << std::setprecision(3) << (hash_rate / 1e9) << " GH/s"
                        << std::endl;

                if (blocks_found > 0) {
                    double blocks_per_second = (double) blocks_found / (elapsed / 1000.0);
                    std::cout << "Blocks found rate: " << std::fixed << std::setprecision(2) << blocks_per_second <<
                            " blocks/s" << std::endl;

                    // Estimate completion time
                    if (blocks_per_second > 0) {
                        double remaining_blocks = target_blocks - blocks_found;
                        double eta_seconds = remaining_blocks / blocks_per_second;
                        std::cout << "Estimated time to complete Phase 1: " << std::fixed << std::setprecision(1) << (
                            eta_seconds / 60.0) << " minutes" << std::endl;
                    }
                }
            }
            std::cout << std::string(60, '-') << std::endl;
        }
    }

    std::cout << "\nPhase 1 complete! Found " << blocks_found << " near-collision blocks." << std::endl;

    // ==================== Phase 2: Birthday Attack ====================

    std::cout << "\n=== Phase 2: Birthday Attack for Full Collision ===" << std::endl;

    // Allocate memory for collision output
    uint32_t *d_collision_msg1, *d_collision_msg2;
    uint32_t *d_found_flag;

    checkCudaError(cudaMalloc(&d_collision_msg1, 32 * sizeof(uint32_t)), "Allocating collision msg1");
    checkCudaError(cudaMalloc(&d_collision_msg2, 32 * sizeof(uint32_t)), "Allocating collision msg2");
    checkCudaError(cudaMalloc(&d_found_flag, sizeof(uint32_t)), "Allocating found flag");

    checkCudaError(cudaMemset(d_found_flag, 0, sizeof(uint32_t)), "Initializing found flag");

    // Launch birthday attack
    int birthday_blocks = std::min(max_blocks * 2, 512);

    std::cout << "Launching birthday attack with " << birthday_blocks << " blocks..." << std::endl;
    std::cout << "Searching through " << blocks_found << " near-collision blocks..." << std::endl;

    auto birthday_start = std::chrono::high_resolution_clock::now();

    birthday_attack<<<birthday_blocks, THREADS_PER_BLOCK>>>(
        d_blocks,
        blocks_found,
        d_collision_msg1,
        d_collision_msg2,
        d_found_flag
    );

    checkCudaError(cudaGetLastError(), "Launching birthday attack");
    checkCudaError(cudaDeviceSynchronize(), "Birthday attack completion");

    auto birthday_end = std::chrono::high_resolution_clock::now();
    auto birthday_time = std::chrono::duration_cast<std::chrono::seconds>(birthday_end - birthday_start).count();
    std::cout << "Birthday attack completed in " << birthday_time << " seconds." << std::endl;

    // Check if collision was found
    uint32_t found_flag;
    checkCudaError(cudaMemcpy(&found_flag, d_found_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                   "Reading found flag");

    if (found_flag) {
        std::cout << "\n*** COLLISION FOUND! ***" << std::endl;

        // Copy collision messages to host
        uint32_t h_collision_msg1[32], h_collision_msg2[32];
        checkCudaError(cudaMemcpy(h_collision_msg1, d_collision_msg1, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                       "Reading collision msg1");
        checkCudaError(cudaMemcpy(h_collision_msg2, d_collision_msg2, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                       "Reading collision msg2");

        // Convert to bytes for saving
        uint8_t msg1_bytes[128], msg2_bytes[128];
        for (int i = 0; i < 32; i++) {
            // Convert uint32_t to big-endian bytes
            msg1_bytes[i * 4] = (h_collision_msg1[i] >> 24) & 0xFF;
            msg1_bytes[i * 4 + 1] = (h_collision_msg1[i] >> 16) & 0xFF;
            msg1_bytes[i * 4 + 2] = (h_collision_msg1[i] >> 8) & 0xFF;
            msg1_bytes[i * 4 + 3] = h_collision_msg1[i] & 0xFF;

            msg2_bytes[i * 4] = (h_collision_msg2[i] >> 24) & 0xFF;
            msg2_bytes[i * 4 + 1] = (h_collision_msg2[i] >> 16) & 0xFF;
            msg2_bytes[i * 4 + 2] = (h_collision_msg2[i] >> 8) & 0xFF;
            msg2_bytes[i * 4 + 3] = h_collision_msg2[i] & 0xFF;
        }

        // Save collision to files
        std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string filename1 = "collision_" + timestamp + "_msg1.bin";
        std::string filename2 = "collision_" + timestamp + "_msg2.bin";

        std::ofstream file1(filename1, std::ios::binary);
        std::ofstream file2(filename2, std::ios::binary);

        if (file1.is_open() && file2.is_open()) {
            file1.write(reinterpret_cast<const char *>(msg1_bytes), 128);
            file2.write(reinterpret_cast<const char *>(msg2_bytes), 128);

            std::cout << "\nCollision saved to:" << std::endl;
            std::cout << "  " << filename1 << std::endl;
            std::cout << "  " << filename2 << std::endl;
        }

        file1.close();
        file2.close();

        // Display first few bytes
        std::cout << "\nFirst 32 bytes of message 1: ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << h_collision_msg1[i] << " ";
        }
        std::cout << std::dec << std::endl;

        std::cout << "First 32 bytes of message 2: ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << h_collision_msg2[i] << " ";
        }
        std::cout << std::dec << std::endl;
    } else {
        std::cout << "\nNo collision found in this run. More blocks may be needed." << std::endl;
        std::cout << "Try increasing the target blocks with -b option." << std::endl;
    }

    // ==================== Cleanup ====================

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "\nTotal execution time: " << total_elapsed << " seconds" << std::endl;
    std::cout << "Total attempts: " << total_attempts << " (2^" << std::fixed << std::setprecision(1) <<
            log2((double) total_attempts) << ")" << std::endl;

    // Calculate overall performance
    if (total_elapsed > 0) {
        double overall_hash_rate = (double) total_attempts / total_elapsed;
        std::cout << "Overall hash rate: " << std::fixed << std::setprecision(3) << (overall_hash_rate / 1e9) << " GH/s"
                << std::endl;
    }

    // Free device memory
    cudaFree(d_blocks);
    cudaFree(d_block_count);
    cudaFree(d_best_quality);
    cudaFree(d_collision_msg1);
    cudaFree(d_collision_msg2);
    cudaFree(d_found_flag);

    cudaDeviceReset();

    return found_flag ? 0 : 1;
}
