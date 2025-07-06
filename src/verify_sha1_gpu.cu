#include "sha1_miner.cuh"
#include "utilities.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

// Debug kernel to show what's happening with nonce application
__global__ void debug_nonce_kernel(uint64_t nonce, uint32_t *debug_output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Show what __byte_perm does
    uint32_t nonce_high = __byte_perm(static_cast<uint32_t>(nonce >> 32), 0, 0x0123);
    uint32_t nonce_low = __byte_perm(static_cast<uint32_t>(nonce & 0xFFFFFFFF), 0, 0x0123);

    debug_output[0] = static_cast<uint32_t>(nonce >> 32); // Original high
    debug_output[1] = static_cast<uint32_t>(nonce & 0xFFFFFFFF); // Original low
    debug_output[2] = nonce_high; // After byte_perm
    debug_output[3] = nonce_low; // After byte_perm
}

// Corrected CPU implementation that matches GPU behavior
void compute_sha1_with_nonce_cpu_correct(const uint8_t *message, uint64_t nonce, uint8_t *hash_out) {
    std::vector<uint8_t> msg_copy(message, message + 32);

    // The GPU kernel does this:
    // 1. Loads message bytes into 32-bit words in big-endian format
    // 2. Applies __byte_perm to the nonce (which reverses byte order)
    // 3. XORs the reversed nonce with the big-endian words

    // Step 1: Get words 6 and 7 in big-endian
    uint32_t word6 = (msg_copy[24] << 24) | (msg_copy[25] << 16) |
                     (msg_copy[26] << 8) | msg_copy[27];
    uint32_t word7 = (msg_copy[28] << 24) | (msg_copy[29] << 16) |
                     (msg_copy[30] << 8) | msg_copy[31];

    // Step 2: Apply nonce with byte reversal (simulating __byte_perm)
    uint32_t nonce_high_le = static_cast<uint32_t>(nonce >> 32);
    uint32_t nonce_low_le = static_cast<uint32_t>(nonce & 0xFFFFFFFF);

    // __byte_perm(x, 0, 0x0123) reverses the bytes
    uint32_t nonce_high = ((nonce_high_le & 0xFF) << 24) |
                          ((nonce_high_le & 0xFF00) << 8) |
                          ((nonce_high_le & 0xFF0000) >> 8) |
                          ((nonce_high_le & 0xFF000000) >> 24);
    uint32_t nonce_low = ((nonce_low_le & 0xFF) << 24) |
                         ((nonce_low_le & 0xFF00) << 8) |
                         ((nonce_low_le & 0xFF0000) >> 8) |
                         ((nonce_low_le & 0xFF000000) >> 24);

    // Step 3: XOR
    word6 ^= nonce_high;
    word7 ^= nonce_low;

    // Convert back to bytes
    msg_copy[24] = (word6 >> 24) & 0xFF;
    msg_copy[25] = (word6 >> 16) & 0xFF;
    msg_copy[26] = (word6 >> 8) & 0xFF;
    msg_copy[27] = word6 & 0xFF;
    msg_copy[28] = (word7 >> 24) & 0xFF;
    msg_copy[29] = (word7 >> 16) & 0xFF;
    msg_copy[30] = (word7 >> 8) & 0xFF;
    msg_copy[31] = word7 & 0xFF;

    auto hash = calculate_sha1(msg_copy);
    std::memcpy(hash_out, hash.data(), 20);
}


// Also need to fix the kernel definition to match what's in the test
__global__ void test_sha1_kernel(
    const uint8_t *message,
    size_t message_len,
    uint64_t nonce,
    uint32_t *output_hash
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // SHA-1 constants
    const uint32_t K[4] = {0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6};
    const uint32_t H0[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

    // Prepare message
    uint32_t W[16];
    memset(W, 0, sizeof(W));

    // Copy message and convert to big-endian
    for (size_t i = 0; i < message_len && i < 32; i += 4) {
        uint32_t word = 0;
        for (size_t j = 0; j < 4 && i + j < message_len; j++) {
            word |= static_cast<uint32_t>(message[i + j]) << (24 - j * 8);
        }
        W[i / 4] = word;
    }

    // Apply nonce to last 8 bytes (words 6 and 7)
    uint32_t nonce_high = __byte_perm(static_cast<uint32_t>(nonce >> 32), 0, 0x0123);
    uint32_t nonce_low = __byte_perm(static_cast<uint32_t>(nonce & 0xFFFFFFFF), 0, 0x0123);

    W[6] ^= nonce_high;
    W[7] ^= nonce_low;

    // Padding
    W[8] = 0x80000000;
    W[15] = 0x00000100; // Message length in bits (256 in big-endian)

    // Initialize working variables
    uint32_t a = H0[0];
    uint32_t b = H0[1];
    uint32_t c = H0[2];
    uint32_t d = H0[3];
    uint32_t e = H0[4];

    // Main SHA-1 computation
#pragma unroll 4
    for (int t = 0; t < 80; t++) {
        if (t >= 16) {
            uint32_t w_t = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                           W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = (w_t << 1) | (w_t >> 31); // Rotate left by 1
        }

        uint32_t f, k;
        if (t < 20) {
            f = (b & c) | (~b & d);
            k = K[0];
        } else if (t < 40) {
            f = b ^ c ^ d;
            k = K[1];
        } else if (t < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = K[2];
        } else {
            f = b ^ c ^ d;
            k = K[3];
        }

        uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k + W[t & 15];
        e = d;
        d = c;
        c = (b << 30) | (b >> 2);
        b = a;
        a = temp;
    }

    // Add initial values
    output_hash[0] = a + H0[0];
    output_hash[1] = b + H0[1];
    output_hash[2] = c + H0[2];
    output_hash[3] = d + H0[3];
    output_hash[4] = e + H0[4];
}


// Test GPU implementation with debug info
extern "C" bool test_gpu_sha1_implementation() {
    std::cout << "Testing GPU SHA-1 with nonce application (DEBUG VERSION):\n\n";

    // First, let's understand what __byte_perm does
    std::cout << "Understanding __byte_perm behavior:\n";
    std::vector<uint64_t> debug_nonces = {1, 0x123456789ABCDEF0ULL};

    for (uint64_t nonce: debug_nonces) {
        uint32_t *d_debug;
        cudaMalloc(&d_debug, 4 * sizeof(uint32_t));

        debug_nonce_kernel<<<1, 1>>>(nonce, d_debug);
        cudaDeviceSynchronize();

        uint32_t debug_values[4];
        cudaMemcpy(debug_values, d_debug, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        std::cout << "  Nonce: 0x" << std::hex << std::setw(16) << std::setfill('0') << nonce << "\n";
        std::cout << "    Original high: 0x" << std::setw(8) << debug_values[0] << "\n";
        std::cout << "    Original low:  0x" << std::setw(8) << debug_values[1] << "\n";
        std::cout << "    After byte_perm high: 0x" << std::setw(8) << debug_values[2] << "\n";
        std::cout << "    After byte_perm low:  0x" << std::setw(8) << debug_values[3] << "\n\n";

        cudaFree(d_debug);
    }

    // Now test with the corrected implementation
    std::cout << "Testing with corrected CPU implementation:\n";

    // Test message (32 bytes)
    std::vector<uint8_t> message(32, 0);
    std::strcpy(reinterpret_cast<char *>(message.data()), "SHA-1 GPU Test Message");

    // Test with different nonces
    std::vector<uint64_t> test_nonces = {0, 1, 0x123456789ABCDEF0ULL, UINT64_MAX};

    bool all_passed = true;

    for (uint64_t nonce: test_nonces) {
        // Allocate device memory
        uint8_t *d_message;
        uint32_t *d_hash;
        cudaMalloc(&d_message, 32);
        cudaMalloc(&d_hash, 5 * sizeof(uint32_t));

        // Copy message to device
        cudaMemcpy(d_message, message.data(), 32, cudaMemcpyHostToDevice);

        // Launch kernel (using the test kernel from your code)
        test_sha1_kernel<<<1, 1>>>(d_message, 32, nonce, d_hash);
        cudaDeviceSynchronize();

        // Get result
        uint32_t gpu_hash[5];
        cudaMemcpy(gpu_hash, d_hash, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compute expected hash on CPU with corrected implementation
        uint8_t expected_hash[20];
        compute_sha1_with_nonce_cpu_correct(message.data(), nonce, expected_hash);

        // Convert GPU result to bytes for comparison
        std::vector<uint8_t> gpu_hash_bytes(20);
        for (int i = 0; i < 5; i++) {
            gpu_hash_bytes[i * 4] = (gpu_hash[i] >> 24) & 0xFF;
            gpu_hash_bytes[i * 4 + 1] = (gpu_hash[i] >> 16) & 0xFF;
            gpu_hash_bytes[i * 4 + 2] = (gpu_hash[i] >> 8) & 0xFF;
            gpu_hash_bytes[i * 4 + 3] = gpu_hash[i] & 0xFF;
        }

        bool passed = (std::memcmp(gpu_hash_bytes.data(), expected_hash, 20) == 0);

        std::cout << "\n  Nonce: 0x" << std::hex << nonce << std::dec << "\n";
        std::cout << "  GPU Hash: " << bytes_to_hex(gpu_hash_bytes) << "\n";
        std::cout << "  Expected: " << bytes_to_hex(std::vector<uint8_t>(expected_hash, expected_hash + 20)) << "\n";
        std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n";

        all_passed &= passed;

        // Cleanup
        cudaFree(d_message);
        cudaFree(d_hash);
    }

    return all_passed;
}
