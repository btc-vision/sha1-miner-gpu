#include "test_utils.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "sha1_miner.cuh"

#include "../../logging/logger.hpp"
#include "mining_system.hpp"

// Generate cryptographically secure random message
std::vector<uint8_t> generate_secure_random_message()
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<uint8_t> message(32);

    // Use multiple sources of entropy
    auto now   = std::chrono::high_resolution_clock::now();
    auto nanos = now.time_since_epoch().count();

    // Mix in time-based entropy
    for (size_t i = 0; i < 8; i++) {
        message[i] = static_cast<uint8_t>((nanos >> (i * 8)) & 0xFF);
    }

    // Fill rest with random data
    for (size_t i = 8; i < 32; i++) {
        message[i] = static_cast<uint8_t>(dis(gen));
    }

    // Additional mixing
    for (size_t i = 0; i < 32; i++) {
        message[i] ^= static_cast<uint8_t>(dis(gen));
    }

    return message;
}

// Verify SHA-1 implementation
bool verify_sha1_implementation()
{
    LOG_INFO("TEST", "Verifying SHA-1 implementation...");

    // Test vectors from FIPS 180-1
    struct TestVector
    {
        std::string message;
        std::string expected_hash;
    };

    std::vector<TestVector> test_vectors = {
        {"abc",                                                      "a9993e364706816aba3e25717850c26c9cd0d89d"},
        {"",                                                         "da39a3ee5e6b4b0d3255bfef95601890afd80709"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "84983e441c3bd26ebaae4aa1f95129e5e54670f1"}
    };

    bool all_passed = true;

    for (const auto &test : test_vectors) {
        std::vector<uint8_t> message(test.message.begin(), test.message.end());
        auto hash            = calculate_sha1(message);
        std::string hash_hex = bytes_to_hex(hash);

        if (hash_hex != test.expected_hash) {
            LOG_ERROR("TEST", "FAILED: Message '", test.message, "'");
            LOG_ERROR("TEST", "  Expected: ", test.expected_hash);
            LOG_ERROR("TEST", "  Got:      ", hash_hex);
            all_passed = false;
        } else {
            LOG_INFO("TEST", Color::GREEN, "PASSED: '", test.message, "'", Color::RESET);
        }
    }

    return all_passed;
}

// Run comprehensive benchmark
void run_benchmark(const int gpu_id)
{
    LOG_INFO("BENCH", Color::BRIGHT_YELLOW, "=== SHA-1 Near-Collision Mining Benchmark ===", Color::RESET);

    // Initialize mining system with auto-tuned parameters
    MiningSystem::Config sys_config;
    sys_config.device_id = gpu_id;

    g_mining_system = std::make_unique<MiningSystem>(sys_config);
    if (!g_mining_system->initialize()) {
        LOG_ERROR("BENCH", "Failed to initialize mining system");
        return;
    }

    // Test different difficulty levels
    const std::vector<uint32_t> difficulties = {60};
    std::vector<double> results;

    for (const uint32_t diff : difficulties) {
        if (g_shutdown)
            break;

        LOG_INFO("BENCH", "Testing difficulty ", diff, " bits:");

        // Generate test job
        auto message     = generate_secure_random_message();
        auto target_hash = calculate_sha1(message);

        MiningJob job = create_mining_job(message.data(), target_hash.data(), diff);

        auto start = std::chrono::steady_clock::now();
        g_mining_system->runMiningLoop(job);
        auto end = std::chrono::steady_clock::now();

        // Get statistics
        auto stats       = g_mining_system->getStats();
        double hash_rate = stats.hash_rate / 1e9;  // GH/s

        results.push_back(hash_rate);

        LOG_INFO("BENCH", "Results for difficulty ", diff, " bits:");
        LOG_INFO("BENCH", "  Hash rate: ", std::fixed, std::setprecision(2), hash_rate, " GH/s");
        LOG_INFO("BENCH", "  Candidates found: ", stats.candidates_found);
        LOG_INFO("BENCH", "  Expected candidates: ", std::scientific, (stats.hashes_computed / std::pow(2.0, diff)));
        LOG_INFO("BENCH", "  Efficiency: ", std::fixed, std::setprecision(2),
                 (100.0 * stats.candidates_found * std::pow(2.0, diff) / stats.hashes_computed), "%");
    }

    // Print summary
    LOG_INFO("BENCH", Color::BRIGHT_GREEN, "=== Benchmark Summary ===", Color::RESET);
    LOG_INFO("BENCH", "Difficulty | Hash Rate (GH/s)");
    LOG_INFO("BENCH", "-----------|----------------");
    for (size_t i = 0; i < difficulties.size() && i < results.size(); i++) {
        LOG_INFO("BENCH", std::setw(10), difficulties[i], " | ", std::fixed, std::setprecision(2), results[i]);
    }

    cleanup_mining_system();
}
