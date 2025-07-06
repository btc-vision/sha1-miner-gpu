// verify_sha1_gpu.cu - GPU-specific SHA-1 tests

#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// Forward declaration from kernel
extern __device__ void sha1_compute(const uint32_t*, uint64_t, uint32_t*);

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

// GPU kernel for testing single SHA-1 computation
__global__ void test_sha1_kernel(
    const uint8_t *message,
    size_t message_len,
    uint64_t nonce,
    uint32_t *output_hash
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Prepare message (pad to 32 bytes for our mining format)
    uint32_t padded_msg[8] = {0};

    // Copy message and convert to big-endian
    for (size_t i = 0; i < message_len && i < 32; i += 4) {
        uint32_t word = 0;
        for (size_t j = 0; j < 4 && i + j < message_len; j++) {
            word |= static_cast<uint32_t>(message[i + j]) << (24 - j * 8);
        }
        padded_msg[i / 4] = word;
    }

    // Compute SHA-1 with nonce
    sha1_compute(padded_msg, nonce, output_hash);
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