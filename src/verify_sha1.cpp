#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cassert>
#include <chrono>
#include <sstream>
#include <ctime>
#include "cxxsha1.hpp"

// Test vectors from NIST
struct TestVector {
    const char* message;
    const char* expected_hash;
};

const TestVector test_vectors[] = {
    // Standard test vectors
    {"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"},
    {"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"},
    {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
     "84983e441c3bd26ebaae4aa1f95129e5e54670f1"},
    {"The quick brown fox jumps over the lazy dog",
     "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"},

    // 32-byte messages (our use case)
    {"0123456789abcdef0123456789abcdef", "8c0ae11688016515c088b8419513ae7fb0b8ee88"},
    {"ffffffffffffffffffffffffffffffff", "c907254fba426c1c7e46b0bb89cefc7b43aef714"},
};

// Convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        bytes.push_back(std::stoi(byte_str, nullptr, 16));
    }
    return bytes;
}

// Convert bytes to hex string
std::string bytes_to_hex(const uint8_t* bytes, size_t len) {
    std::stringstream ss;
    for (size_t i = 0; i < len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
    }
    return ss.str();
}

// Count matching bits between two hashes
int count_matching_bits(const uint8_t* hash1, const uint8_t* hash2) {
    int matching_bits = 0;

    for (int i = 0; i < 20; i++) {
        uint8_t xor_byte = hash1[i] ^ hash2[i];
        // Use popcount to count set bits in XOR result
        matching_bits += 8 - __builtin_popcount(static_cast<unsigned int>(xor_byte));
    }

    return matching_bits;
}

// Count consecutive matching bits from MSB
int count_consecutive_bits(const uint8_t* hash1, const uint8_t* hash2) {
    int consecutive_bits = 0;

    for (int i = 0; i < 20; i++) {
        uint8_t xor_byte = hash1[i] ^ hash2[i];

        if (xor_byte == 0) {
            consecutive_bits += 8;
        } else {
            // Count leading zeros in the byte
            consecutive_bits += __builtin_clz(static_cast<unsigned int>(xor_byte)) - 24; // Adjust for byte
            break;
        }
    }

    return consecutive_bits;
}

// Test basic SHA-1 functionality
bool test_sha1_basic() {
    std::cout << "Testing SHA-1 implementation...\n\n";

    bool all_passed = true;

    for (const auto& tv : test_vectors) {
        SHA1 sha1;
        sha1.update(std::string(tv.message));
        std::string hash_hex = sha1.final();

        bool passed = (hash_hex == tv.expected_hash);

        std::cout << "Message: \"" << tv.message << "\"\n";
        std::cout << "Expected: " << tv.expected_hash << "\n";
        std::cout << "Got:      " << hash_hex << "\n";
        std::cout << "Status:   " << (passed ? "PASS" : "FAIL") << "\n\n";

        all_passed &= passed;
    }

    return all_passed;
}

// Test near-collision detection
void test_near_collision() {
    std::cout << "Testing near-collision detection...\n\n";

    // Create two similar messages
    uint8_t msg1[32] = {0};
    uint8_t msg2[32] = {0};

    // Make them slightly different
    msg2[31] = 1;  // Change last byte

    // Compute hashes using SHA1 class
    SHA1 sha1_1, sha1_2;
    sha1_1.update(std::string(reinterpret_cast<char*>(msg1), 32));
    sha1_2.update(std::string(reinterpret_cast<char*>(msg2), 32));

    std::string hex1 = sha1_1.final();
    std::string hex2 = sha1_2.final();

    // Convert hex to binary
    uint8_t hash1[20], hash2[20];
    for (int i = 0; i < 20; i++) {
        hash1[i] = static_cast<uint8_t>(std::stoi(hex1.substr(i * 2, 2), nullptr, 16));
        hash2[i] = static_cast<uint8_t>(std::stoi(hex2.substr(i * 2, 2), nullptr, 16));
    }

    // Analyze similarity
    int matching_bits = count_matching_bits(hash1, hash2);
    int consecutive_bits = count_consecutive_bits(hash1, hash2);

    std::cout << "Message 1: " << bytes_to_hex(msg1, 32) << "\n";
    std::cout << "Hash 1:    " << bytes_to_hex(hash1, 20) << "\n\n";

    std::cout << "Message 2: " << bytes_to_hex(msg2, 32) << "\n";
    std::cout << "Hash 2:    " << bytes_to_hex(hash2, 20) << "\n\n";

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

// Simulate mining attempts
void simulate_mining() {
    std::cout << "\nSimulating mining attempts...\n\n";

    // Fixed base message
    uint8_t base_msg[32];
    for (int i = 0; i < 32; i++) {
        base_msg[i] = static_cast<uint8_t>(i);
    }

    // Compute target hash
    SHA1 sha1_target;
    sha1_target.update(std::string(reinterpret_cast<char*>(base_msg), 32));
    std::string target_hex = sha1_target.final();

    uint8_t target_hash[20];
    for (int i = 0; i < 20; i++) {
        target_hash[i] = static_cast<uint8_t>(std::stoi(target_hex.substr(i * 2, 2), nullptr, 16));
    }

    std::cout << "Target: " << bytes_to_hex(target_hash, 20) << "\n\n";

    // Try different nonces
    int best_match = 0;
    uint64_t best_nonce = 0;

    const uint64_t num_attempts = 1000000;

    for (uint64_t nonce = 0; nonce < num_attempts; nonce++) {
        // Apply nonce to message
        uint8_t msg[32];
        std::memcpy(msg, base_msg, 32);

        // XOR nonce into last 8 bytes
        *reinterpret_cast<uint64_t*>(&msg[24]) ^= nonce;

        // Compute hash
        SHA1 sha1;
        sha1.update(std::string(reinterpret_cast<char*>(msg), 32));
        std::string hex = sha1.final();

        uint8_t hash[20];
        for (int i = 0; i < 20; i++) {
            hash[i] = static_cast<uint8_t>(std::stoi(hex.substr(i * 2, 2), nullptr, 16));
        }

        // Check similarity
        int matching_bits = count_matching_bits(hash, target_hash);

        if (matching_bits > best_match) {
            best_match = matching_bits;
            best_nonce = nonce;

            if (matching_bits >= 20) {  // Found something interesting
                std::cout << "Nonce: " << std::hex << nonce << std::dec
                          << " | Matching bits: " << matching_bits
                          << " | Hash: " << bytes_to_hex(hash, 20) << "\n";
            }
        }

        if (nonce % 100000 == 0) {
            std::cout << "\rProgress: " << (nonce * 100 / num_attempts) << "%"
                      << " | Best: " << best_match << " bits" << std::flush;
        }
    }

    std::cout << "\n\nBest result after " << num_attempts << " attempts:\n";
    std::cout << "Nonce: 0x" << std::hex << best_nonce << std::dec << "\n";
    std::cout << "Matching bits: " << best_match << " / 160\n";
}

// Performance test
void performance_test() {
    std::cout << "\nPerformance test...\n";

    const int num_hashes = 1000000;
    uint8_t msg[32] = {0};
    uint8_t hash[20];

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_hashes; i++) {
        // Vary the message
        *reinterpret_cast<uint32_t*>(msg) = static_cast<uint32_t>(i);

        SHA1 sha1;
        sha1.update(std::string(reinterpret_cast<char*>(msg), 32));
        std::string result = sha1.final();
        // Just computing, not storing
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double seconds = duration.count() / 1000000.0;
    double hashes_per_second = num_hashes / seconds;

    std::cout << "Computed " << num_hashes << " hashes in "
              << std::fixed << std::setprecision(3) << seconds << " seconds\n";
    std::cout << "Rate: " << std::fixed << std::setprecision(2)
              << hashes_per_second / 1000000.0 << " MH/s (CPU)\n";
}

int main() {
    std::cout << "+------------------------------------------+\n";
    std::cout << "|       SHA-1 Implementation Test          |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Run tests
    if (!test_sha1_basic()) {
        std::cerr << "SHA-1 basic tests FAILED!\n";
        return 1;
    }

    test_near_collision();
    simulate_mining();
    performance_test();

    std::cout << "\nAll tests completed successfully!\n";

    return 0;
}