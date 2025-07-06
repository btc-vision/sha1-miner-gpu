#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cassert>
#include <chrono>
#include <sstream>
#include "sha1_miner_functions.h"
#include "cxxsha1.hpp"
#include "sha1_miner.cuh"
#include <cuda_runtime.h>

// Cross-platform bit manipulation functions
#ifdef _MSC_VER
#include <intrin.h>
// MSVC implementations
inline int popcount(unsigned int x) {
    return __popcnt(x);
}

inline int count_leading_zeros(unsigned int x) {
    unsigned long index;
    if (_BitScanReverse(&index, x)) {
        return 31 - index;
    }
    return 32; // All bits are zero
}
#else
// GCC/Clang implementations
inline int popcount(unsigned int x) {
    return __builtin_popcount(x);
}

inline int count_leading_zeros(unsigned int x) {
    return x ? __builtin_clz(x) : 32;
}
#endif

// Test vectors from NIST
struct TestVector {
    const char *message;
    const char *expected_hash;
    bool is_hex_input; // true if message should be interpreted as hex
};

const TestVector test_vectors[] = {
    // Standard test vectors (ASCII strings)
    {"", "da39a3ee5e6b4b0d3255bfef95601890afd80709", false},
    {"abc", "a9993e364706816aba3e25717850c26c9cd0d89d", false},
    {
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1", false
    },
    {
        "The quick brown fox jumps over the lazy dog",
        "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12", false
    },

    // 32-byte binary messages (hex input)
    {"0123456789abcdef0123456789abcdef", "4cc4cf5b00c0e2f9c2d91768e9ca5def474c6908", true},
    {
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "c8ae0c9a79e2aee3a1b807103187a9ec51c59d3e", true
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000000",
        "07bae34777fe7569d8dd66701fb23a0cd4775d02", true
    },
};

// Convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        bytes.push_back(static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16)));
    }
    return bytes;
}

// Convert bytes to hex string
std::string bytes_to_hex(const uint8_t *bytes, size_t len) {
    std::stringstream ss;
    for (size_t i = 0; i < len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
    }
    return ss.str();
}

// Count matching bits between two hashes
int count_matching_bits(const uint8_t *hash1, const uint8_t *hash2) {
    int matching_bits = 0;

    for (int i = 0; i < 20; i++) {
        uint8_t xor_byte = hash1[i] ^ hash2[i];
        // Use popcount to count set bits in XOR result
        matching_bits += 8 - popcount(static_cast<unsigned int>(xor_byte));
    }

    return matching_bits;
}

// Count consecutive matching bits from MSB
int count_consecutive_bits(const uint8_t *hash1, const uint8_t *hash2) {
    int consecutive_bits = 0;

    for (int i = 0; i < 20; i++) {
        uint8_t xor_byte = hash1[i] ^ hash2[i];

        if (xor_byte == 0) {
            consecutive_bits += 8;
        } else {
            // Count leading zeros in the byte
            consecutive_bits += count_leading_zeros(static_cast<unsigned int>(xor_byte)) - 24; // Adjust for byte
            break;
        }
    }

    return consecutive_bits;
}

// Test basic SHA-1 functionality
bool test_sha1_basic() {
    std::cout << "Testing SHA-1 implementation...\n\n";

    bool all_passed = true;

    for (const auto &tv: test_vectors) {
        SHA1 sha1;

        if (tv.is_hex_input) {
            // Convert hex string to binary data
            auto binary_data = hex_to_bytes(tv.message);
            sha1.update(std::string(reinterpret_cast<char *>(binary_data.data()), binary_data.size()));
        } else {
            // Use string as-is
            sha1.update(std::string(tv.message));
        }

        std::string hash_hex = sha1.final();

        bool passed = (hash_hex == tv.expected_hash);

        std::cout << "Message: \"" << tv.message << "\""
                << (tv.is_hex_input ? " (hex)" : " (ascii)") << "\n";
        std::cout << "Expected: " << tv.expected_hash << "\n";
        std::cout << "Got:      " << hash_hex << "\n";
        std::cout << "Status:   " << (passed ? "PASS" : "FAIL") << "\n\n";

        all_passed &= passed;
    }

    return all_passed;
}

// Test GPU mining functionality
void test_gpu_mining() {
    std::cout << "\n=== Testing GPU Mining Functions ===\n\n";

    // Check for CUDA devices
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA devices found. Skipping GPU tests.\n";
        return;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)\n";

    // Test 1: Initialize mining system
    std::cout << "\n1. Testing mining system initialization...\n";
    if (!init_mining_system(0)) {
        std::cerr << "Failed to initialize mining system\n";
        return;
    }
    std::cout << "✓ Mining system initialized successfully\n";

    // Test 2: Create a mining job
    std::cout << "\n2. Testing mining job creation...\n";
    uint8_t test_message[32] = {0};
    for (int i = 0; i < 32; i++) {
        test_message[i] = static_cast<uint8_t>(i);
    }

    // Calculate target hash
    SHA1 sha1;
    sha1.update(std::string(reinterpret_cast<char *>(test_message), 32));
    std::string target_hex = sha1.final();
    auto target_bytes = hex_to_bytes(target_hex);

    MiningJob job = create_mining_job(test_message, target_bytes.data(), 20); // Low difficulty for testing
    std::cout << "✓ Mining job created\n";
    std::cout << "  Base message: " << bytes_to_hex(test_message, 32) << "\n";
    std::cout << "  Target hash:  " << target_hex << "\n";
    std::cout << "  Difficulty:   " << job.difficulty << " bits\n";

    // Test 3: Run a short mining test
    std::cout << "\n3. Running GPU mining for 5 seconds...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run mining for 5 seconds
    run_mining_loop(job, 5);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\n✓ GPU mining completed in " << duration.count() << " seconds\n";

    // Test 4: Compare GPU vs CPU performance
    std::cout << "\n4. Performance comparison (1 million hashes)...\n";

    // CPU performance
    const int cpu_hashes = 1000000;
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < cpu_hashes; i++) {
        uint8_t msg[32];
        std::memcpy(msg, test_message, 32);
        *reinterpret_cast<uint64_t *>(&msg[24]) ^= i;

        SHA1 sha1_cpu;
        sha1_cpu.update(std::string(reinterpret_cast<char *>(msg), 32));
        sha1_cpu.final();
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    double cpu_rate = static_cast<double>(cpu_hashes) / (cpu_duration.count() / 1000000.0) / 1000000.0;

    std::cout << "  CPU Rate: " << std::fixed << std::setprecision(2) << cpu_rate << " MH/s\n";
    std::cout << "  GPU Rate: Check the mining output above for GH/s rate\n";
    std::cout << "  (GPU is typically 100-1000x faster than CPU)\n";

    // Test 5: Test result processing
    std::cout << "\n5. Testing near-collision detection on GPU...\n";

    // Create a harder job that might find near-collisions
    MiningJob hard_job = create_mining_job(test_message, target_bytes.data(), 16); // 16-bit collision
    std::cout << "  Looking for 16-bit near-collisions...\n";

    // Run for 10 seconds
    run_mining_loop(hard_job, 10);

    // Cleanup
    std::cout << "\n6. Cleaning up GPU resources...\n";
    cleanup_mining_system();
    std::cout << "✓ GPU resources cleaned up successfully\n";
}

// Test near-collision detection
void test_near_collision() {
    std::cout << "\nTesting near-collision detection...\n\n";

    // Create two similar messages
    uint8_t msg1[32] = {0};
    uint8_t msg2[32] = {0};

    // Make them slightly different
    msg2[31] = 1; // Change last byte

    // Compute hashes using SHA1 class
    SHA1 sha1_1, sha1_2;
    sha1_1.update(std::string(reinterpret_cast<char *>(msg1), 32));
    sha1_2.update(std::string(reinterpret_cast<char *>(msg2), 32));

    std::string hex1 = sha1_1.final();
    std::string hex2 = sha1_2.final();

    // Convert hex to binary
    auto hash1 = hex_to_bytes(hex1);
    auto hash2 = hex_to_bytes(hex2);

    // Analyze similarity
    int matching_bits = count_matching_bits(hash1.data(), hash2.data());
    int consecutive_bits = count_consecutive_bits(hash1.data(), hash2.data());

    std::cout << "Message 1: " << bytes_to_hex(msg1, 32) << "\n";
    std::cout << "Hash 1:    " << hex1 << "\n\n";

    std::cout << "Message 2: " << bytes_to_hex(msg2, 32) << "\n";
    std::cout << "Hash 2:    " << hex2 << "\n\n";

    std::cout << "Matching bits:     " << matching_bits << " / 160\n";
    std::cout << "Consecutive bits:  " << consecutive_bits << "\n";
    std::cout << "Hamming distance:  " << (160 - matching_bits) << "\n\n";

    // Show bit-by-bit comparison
    std::cout << "Bit-by-bit XOR:\n";
    for (int i = 0; i < 20; i++) {
        uint8_t xor_byte = hash1[i] ^ hash2[i];
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(xor_byte) << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << std::dec << "\n";
}

// Simulate CPU mining attempts
void simulate_cpu_mining() {
    std::cout << "\nSimulating CPU mining attempts...\n\n";

    // Fixed base message
    uint8_t base_msg[32];
    for (int i = 0; i < 32; i++) {
        base_msg[i] = static_cast<uint8_t>(i);
    }

    // Compute target hash
    SHA1 sha1_target;
    sha1_target.update(std::string(reinterpret_cast<char *>(base_msg), 32));
    std::string target_hex = sha1_target.final();
    auto target_hash = hex_to_bytes(target_hex);

    std::cout << "Target: " << target_hex << "\n\n";

    // Try different nonces
    int best_match = 0;
    uint64_t best_nonce = 0;

    const uint64_t num_attempts = 1000000;

    for (uint64_t nonce = 0; nonce < num_attempts; nonce++) {
        // Apply nonce to message
        uint8_t msg[32];
        std::memcpy(msg, base_msg, 32);

        // XOR nonce into last 8 bytes
        *reinterpret_cast<uint64_t *>(&msg[24]) ^= nonce;

        // Compute hash
        SHA1 sha1;
        sha1.update(std::string(reinterpret_cast<char *>(msg), 32));
        std::string hex = sha1.final();
        auto hash = hex_to_bytes(hex);

        // Check similarity
        int matching_bits = count_matching_bits(hash.data(), target_hash.data());

        if (matching_bits > best_match) {
            best_match = matching_bits;
            best_nonce = nonce;

            if (matching_bits >= 20) {
                // Found something interesting
                std::cout << "Nonce: " << std::hex << nonce << std::dec
                        << " | Matching bits: " << matching_bits
                        << " | Hash: " << hex << "\n";
            }
        }

        if (nonce % 100000 == 0) {
            std::cout << "\rProgress: " << static_cast<int>(nonce * 100 / num_attempts) << "%"
                    << " | Best: " << best_match << " bits" << std::flush;
        }
    }

    std::cout << "\n\nBest result after " << num_attempts << " attempts:\n";
    std::cout << "Nonce: 0x" << std::hex << best_nonce << std::dec << "\n";
    std::cout << "Matching bits: " << best_match << " / 160\n";
}

// Performance test
void performance_test() {
    std::cout << "\nCPU Performance test...\n";

    const int num_hashes = 1000000;
    uint8_t msg[32] = {0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_hashes; i++) {
        // Vary the message
        *reinterpret_cast<uint32_t *>(msg) = static_cast<uint32_t>(i);

        SHA1 sha1;
        sha1.update(std::string(reinterpret_cast<char *>(msg), 32));
        std::string result = sha1.final();
        // Just computing, not storing
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double seconds = static_cast<double>(duration.count()) / 1000000.0;
    double hashes_per_second = static_cast<double>(num_hashes) / seconds;

    std::cout << "Computed " << num_hashes << " hashes in "
            << std::fixed << std::setprecision(3) << seconds << " seconds\n";
    std::cout << "Rate: " << std::fixed << std::setprecision(2)
            << hashes_per_second / 1000000.0 << " MH/s (CPU)\n";
}

int main() {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|    SHA-1 Implementation & GPU Test       |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Run CPU tests
    if (!test_sha1_basic()) {
        std::cerr << "SHA-1 basic tests FAILED!\n";
        return 1;
    }

    test_near_collision();
    simulate_cpu_mining();
    performance_test();

    // Run GPU tests
    test_gpu_mining();

    std::cout << "\nAll tests completed!\n";

    return 0;
}
