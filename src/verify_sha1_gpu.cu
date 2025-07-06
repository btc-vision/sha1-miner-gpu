#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

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

// Simple SHA-1 test kernel - self-contained implementation for testing
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

    // Prepare message (pad to 32 bytes for our mining format)
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

    // Apply nonce to last 8 bytes
    W[6] ^= __byte_perm(static_cast<uint32_t>(nonce >> 32), 0, 0x0123);
    W[7] ^= __byte_perm(static_cast<uint32_t>(nonce & 0xFFFFFFFF), 0, 0x0123);

    // Padding
    W[8] = 0x80000000;
    W[15] = 256; // Message length in bits

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

// Test GPU implementation - exported function
extern "C" bool test_gpu_sha1_implementation() {
    std::cout << "Testing GPU SHA-1 with nonce application:\n";

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

        // Launch kernel
        test_sha1_kernel<<<1, 1>>>(d_message, 32, nonce, d_hash);

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_message);
            cudaFree(d_hash);
            return false;
        }

        cudaDeviceSynchronize();

        // Get result
        uint32_t gpu_hash[5];
        cudaMemcpy(gpu_hash, d_hash, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compute expected hash on CPU
        std::vector<uint8_t> cpu_message = message;
        // Apply nonce to last 8 bytes
        uint64_t *nonce_ptr = reinterpret_cast<uint64_t *>(&cpu_message[24]);
        *nonce_ptr ^= nonce;

        SHA1 sha1;
        sha1.update(std::string(cpu_message.begin(), cpu_message.end()));
        std::string hex_result = sha1.final();
        std::vector<uint8_t> expected_hash = hex_to_bytes(hex_result);

        // Convert GPU result to bytes for comparison
        std::vector<uint8_t> gpu_hash_bytes(20);
        for (int i = 0; i < 5; i++) {
            gpu_hash_bytes[i * 4] = (gpu_hash[i] >> 24) & 0xFF;
            gpu_hash_bytes[i * 4 + 1] = (gpu_hash[i] >> 16) & 0xFF;
            gpu_hash_bytes[i * 4 + 2] = (gpu_hash[i] >> 8) & 0xFF;
            gpu_hash_bytes[i * 4 + 3] = gpu_hash[i] & 0xFF;
        }

        bool passed = (gpu_hash_bytes == expected_hash);

        std::cout << "  Nonce: 0x" << std::hex << nonce << std::dec << "\n";
        std::cout << "  GPU Hash: ";
        print_hash(gpu_hash_bytes.data());
        std::cout << "\n";
        std::cout << "  Expected: ";
        print_hash(expected_hash.data());
        std::cout << "\n";
        std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << "\n\n";

        all_passed &= passed;

        // Cleanup
        cudaFree(d_message);
        cudaFree(d_hash);
    }

    return all_passed;
}
