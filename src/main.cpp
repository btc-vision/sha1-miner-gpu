#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <random>
#include <signal.h>

// Global flag for graceful shutdown
volatile bool g_shutdown = false;

// Forward declarations
void run_mining_loop(MiningJob job, uint32_t duration_seconds);

void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down...\n";
    g_shutdown = true;
}

// Parse command line arguments
struct Config {
    int gpu_id = 0;
    uint32_t difficulty = 120;  // Default: 120 bits must match
    uint32_t duration = 60;     // Default: 60 seconds
    bool benchmark = false;
    std::string target_hex;
    std::string message_hex;
};

Config parse_args(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--difficulty" && i + 1 < argc) {
            config.difficulty = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            config.duration = std::stoi(argv[++i]);
        } else if (arg == "--target" && i + 1 < argc) {
            config.target_hex = argv[++i];
        } else if (arg == "--message" && i + 1 < argc) {
            config.message_hex = argv[++i];
        } else if (arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--help") {
            std::cout << "SHA-1 Near-Collision Miner\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --gpu <id>          GPU device ID (default: 0)\n";
            std::cout << "  --difficulty <bits> Number of bits that must match (default: 120)\n";
            std::cout << "  --duration <sec>    Mining duration in seconds (default: 60)\n";
            std::cout << "  --target <hex>      Target hash in hex (40 chars)\n";
            std::cout << "  --message <hex>     Base message in hex (64 chars)\n";
            std::cout << "  --benchmark         Run performance benchmark\n";
            std::cout << "  --help              Show this help\n\n";
            std::cout << "Example:\n";
            std::cout << "  " << argv[0] << " --gpu 0 --difficulty 100 --duration 300\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            std::exit(1);
        }
    }

    return config;
}

// Convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;

    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        bytes.push_back(std::stoi(byte_str, nullptr, 16));
    }

    return bytes;
}

// Generate random message for testing
std::vector<uint8_t> generate_random_message() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<uint8_t> message(32);
    for (auto& byte : message) {
        byte = dis(gen);
    }

    return message;
}

// Calculate SHA-1 hash
std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t>& message) {
    sha1_ctx ctx;
    sha1_init(ctx);
    sha1_update(ctx, message.data(), message.size());

    std::vector<uint8_t> hash(20);
    sha1_final(ctx, hash.data());

    return hash;
}

// Run benchmark mode
void run_benchmark(int gpu_id) {
    std::cout << "\n=== SHA-1 Near-Collision Mining Benchmark ===\n\n";

    if (!init_mining_system(gpu_id)) {
        std::cerr << "Failed to initialize mining system\n";
        return;
    }

    // Test different difficulty levels
    std::vector<uint32_t> difficulties = {80, 90, 100, 110, 120, 130};

    for (uint32_t diff : difficulties) {
        std::cout << "\nTesting difficulty " << diff << " bits:\n";

        // Create test job
        auto message = generate_random_message();
        auto target = calculate_sha1(message);

        MiningJob job = create_mining_job(message.data(), target.data(), diff);

        // Run for 30 seconds
        run_mining_loop(job, 30);
    }

    cleanup_mining_system();
}

// Main program
int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Parse command line
    Config config = parse_args(argc, argv);

    std::cout << "+------------------------------------------+\n";
    std::cout << "|    SHA-1 Near-Collision Miner v2.0       |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Run benchmark if requested
    if (config.benchmark) {
        run_benchmark(config.gpu_id);
        return 0;
    }

    // Initialize mining system
    if (!init_mining_system(config.gpu_id)) {
        std::cerr << "Failed to initialize mining system\n";
        return 1;
    }

    // Prepare message and target
    std::vector<uint8_t> message;
    std::vector<uint8_t> target;

    if (!config.message_hex.empty()) {
        if (config.message_hex.length() != 64) {
            std::cerr << "Message must be 64 hex characters (32 bytes)\n";
            cleanup_mining_system();
            return 1;
        }
        message = hex_to_bytes(config.message_hex);
    } else {
        message = generate_random_message();
        std::cout << "Generated random message: ";
        for (uint8_t b : message) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
        }
        std::cout << "\n";
    }

    if (!config.target_hex.empty()) {
        if (config.target_hex.length() != 40) {
            std::cerr << "Target must be 40 hex characters (20 bytes)\n";
            cleanup_mining_system();
            return 1;
        }
        target = hex_to_bytes(config.target_hex);
    } else {
        target = calculate_sha1(message);
        std::cout << "Target SHA-1: ";
        for (uint8_t b : target) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
        }
        std::cout << "\n";
    }

    // Create mining job
    MiningJob job = create_mining_job(message.data(), target.data(), config.difficulty);

    std::cout << "\nMining Configuration:\n";
    std::cout << "Difficulty: " << config.difficulty << " bits must match\n";
    std::cout << "Duration: " << config.duration << " seconds\n";
    std::cout << "Success probability per hash: 2^-" << config.difficulty << "\n\n";

    // Run mining
    run_mining_loop(job, config.duration);

    // Cleanup
    cleanup_mining_system();

    return 0;
}