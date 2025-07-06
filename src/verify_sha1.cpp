// verify_sha1_fixed.cpp - SHA-1 verification tool (CPU tests only)
// GPU tests are moved to a separate CUDA file

#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// Test vectors from NIST FIPS 180-1
struct TestVector {
    std::string name;
    std::vector<uint8_t> message;
    std::vector<uint8_t> expected_hash;
};

// Convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        bytes.push_back(std::stoi(hex.substr(i, 2), nullptr, 16));
    }
    return bytes;
}

// Print hash in hex format
void print_hash(const uint8_t *hash, size_t len = 20) {
    for (size_t i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(hash[i]);
    }
}

// Test CPU implementation
bool test_cpu_sha1(const TestVector &test) {
    std::cout << "Testing CPU SHA-1: " << test.name << "\n";

    // Compute SHA-1 using our implementation
    SHA1 sha1;
    sha1.update(std::string(test.message.begin(), test.message.end()));
    std::string hex_result = sha1.final();

    // Convert hex to bytes
    std::vector<uint8_t> computed_hash = hex_to_bytes(hex_result);

    // Compare with expected
    bool passed = (computed_hash == test.expected_hash);

    std::cout << "  Input (" << test.message.size() << " bytes): ";
    if (test.message.size() <= 64) {
        print_hash(test.message.data(), test.message.size());
    } else {
        std::cout << "...";
    }
    std::cout << "\n";

    std::cout << "  Expected: ";
    print_hash(test.expected_hash.data());
    std::cout << "\n";

    std::cout << "  Computed: ";
    print_hash(computed_hash.data());
    std::cout << "\n";

    std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";

    return passed;
}

// Forward declaration of GPU test function (implemented in verify_sha1_gpu.cu)
extern "C" bool test_gpu_sha1_implementation();

// Portable count leading zeros function
inline uint32_t count_leading_zeros(uint32_t x) {
    if (x == 0) return 32;

    uint32_t n = 0;
    if (x <= 0x0000FFFF) {
        n += 16;
        x <<= 16;
    }
    if (x <= 0x00FFFFFF) {
        n += 8;
        x <<= 8;
    }
    if (x <= 0x0FFFFFFF) {
        n += 4;
        x <<= 4;
    }
    if (x <= 0x3FFFFFFF) {
        n += 2;
        x <<= 2;
    }
    if (x <= 0x7FFFFFFF) { n += 1; }

    return n;
}

// Test near-collision detection
bool test_near_collision_detection() {
    std::cout << "Testing near-collision detection:\n";

    // Create two hashes with known bit differences
    uint32_t hash1[5] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333333};
    uint32_t hash2[5] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333333};

    // Test exact match (160 bits)
    {
        uint32_t matching_bits = 0;
        for (int i = 0; i < 5; i++) {
            uint32_t xor_val = hash1[i] ^ hash2[i];
            if (xor_val == 0) {
                matching_bits += 32;
            } else {
                matching_bits += count_leading_zeros(xor_val);
                break;
            }
        }

        std::cout << "  Exact match test: " << matching_bits << " bits (expected: 160)\n";
        bool passed = (matching_bits == 160);
        std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";

        if (!passed) return false;
    }

    // Test 1-bit difference in last word
    hash2[4] ^= 0x80000000; // Flip MSB of last word
    {
        uint32_t matching_bits = 0;
        for (int i = 0; i < 5; i++) {
            uint32_t xor_val = hash1[i] ^ hash2[i];
            if (xor_val == 0) {
                matching_bits += 32;
            } else {
                matching_bits += count_leading_zeros(xor_val);
                break;
            }
        }

        std::cout << "  1-bit difference test: " << matching_bits << " bits (expected: 128)\n";
        bool passed = (matching_bits == 128);
        std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";

        if (!passed) return false;
    }

    // Test multiple bit differences
    hash2[2] = 0x11110000; // 4 bits different in middle
    {
        uint32_t matching_bits = 0;
        for (int i = 0; i < 5; i++) {
            uint32_t xor_val = hash1[i] ^ hash2[i];
            if (xor_val == 0) {
                matching_bits += 32;
            } else {
                matching_bits += count_leading_zeros(xor_val);
                break;
            }
        }

        std::cout << "  Multi-bit difference test: " << matching_bits << " bits (expected: 64)\n";
        bool passed = (matching_bits == 64);
        std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";

        if (!passed) return false;
    }

    return true;
}

int main() {
    std::cout << "SHA-1 Implementation Verification Tool\n";
    std::cout << "=====================================\n\n";

    // Initialize CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaSetDevice(0);

    // Test vectors
    std::vector<TestVector> test_vectors = {
        {
            "Empty string",
            {},
            hex_to_bytes("da39a3ee5e6b4b0d3255bfef95601890afd80709")
        },
        {
            "abc",
            {'a', 'b', 'c'},
            hex_to_bytes("a9993e364706816aba3e25717850c26c9cd0d89d")
        },
        {
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            std::vector<uint8_t>(
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" + 56
            ),
            hex_to_bytes("84983e441c3bd26ebaae4aa1f95129e5e54670f1")
        },
        {
            "32-byte message for mining test",
            std::vector<uint8_t>(32, 'X'), // 32 'X' characters
            hex_to_bytes("7a0f092061e7cffe645e99fa4719203623f70e46")
        }
    };

    bool all_passed = true;

    // Test CPU implementation
    std::cout << "1. CPU SHA-1 Tests\n";
    std::cout << "==================\n\n";

    for (const auto &test: test_vectors) {
        all_passed &= test_cpu_sha1(test);
    }

    // Test GPU implementation
    std::cout << "2. GPU SHA-1 Tests\n";
    std::cout << "==================\n\n";

    all_passed &= test_gpu_sha1_implementation();

    // Test near-collision detection
    std::cout << "3. Near-Collision Detection Tests\n";
    std::cout << "=================================\n\n";

    all_passed &= test_near_collision_detection();

    // Summary
    std::cout << "Test Summary\n";
    std::cout << "============\n";
    std::cout << "Overall Result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";

    return all_passed ? 0 : 1;
}
